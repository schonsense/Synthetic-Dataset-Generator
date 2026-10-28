"""Exercise the actual run lifecycle without importing the Tk application.

Run ``python -B -m unittest -v test_dataset_lifecycle``. Functions are extracted
from generate.py's AST, as in the existing opening-mode wiring tests. Real
local dataset readers, run manifests, binding records, and the production saver
are used; only GUI widgets, model workers, and user dialogs are stand-ins.
All files live in temporary directories. No display, network, or API is used.
"""

import ast
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
from queue import Empty, Queue
import random
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from dataset_runtime import DatasetRun, has_active_sources
from dataset_settings import apply_schema, new_source
from dataset_sources import SourceSpec, open_source


REPO = Path(__file__).resolve().parent
RUN_CONFIG = {
    'prompts.use_questions_file': True,
    'generation.num_turns': 1,
    'generation.subject_size': 1000,
    'generation.context_size': 3000,
    'api.master_duplication_mode': False,
    'prompts.opening_mode': False,
}


def extracted_functions(filename, names, namespace):
    module = ast.parse((REPO / filename).read_text(encoding='utf-8'), filename=filename)
    selected = [node for node in module.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    missing = set(names) - {node.name for node in selected}
    if missing:
        raise AssertionError('Missing production definitions: {}'.format(sorted(missing)))
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, 'exec'), namespace)


class Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class Widget:
    def __init__(self, *args, **kwargs):
        self.values = dict(kwargs)

    def config(self, **kwargs):
        self.values.update(kwargs)

    def __setitem__(self, key, value):
        self.values[key] = value

    def pack(self, **kwargs):
        pass

    def winfo_exists(self):
        return True

    def winfo_children(self):
        return []


class JoinObservedThread(threading.Thread):
    """Expose the waiter's first join attempt without test timing assumptions."""
    def __init__(self, release):
        super().__init__(target=lambda: release.wait(10), daemon=True)
        self.join_attempted = threading.Event()

    def join(self, timeout=None):
        self.join_attempted.set()
        return super().join(timeout)


class DatasetLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='sdg-lifecycle-')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.state_path = self.directory / 'generation_state.json'
        self.csv_path = self.directory / 'people.csv'
        self.csv_path.write_text('name,setting\nAda,Library\nBen,Garden\nCora,Station\nDee,Workshop\n', encoding='utf-8')
        source = new_source([])
        source.update(location=str(self.csv_path))
        reader = open_source(SourceSpec(source['source_id'], 'csv', str(self.csv_path)))
        try:
            source = apply_schema(source, reader.schema())
        finally:
            reader.close()
        source['column_aliases'] = {'name': 'person', 'setting': 'setting'}
        self.settings = {'sources': [source]}
        self.config = dict(RUN_CONFIG, datasets=deepcopy(self.settings))
        self.config.update({
            'valkey.enabled': False,
            'api.apis': [{'enabled': True, 'url': 'http://fixture.invalid', 'model': 'fixture', 'threads': 1}],
            'api.threads': 1,
        })
        self.calls = []
        self.root = Mock()
        self.root.winfo_exists.return_value = True
        self.state = SimpleNamespace(
            dataset_runtime=None, loaded_dataset_checkpoint=None,
            checkpoint_restore_blocked=False, clearing_job=False, shutting_down=False,
            processing_active=False, stop_processing=False, pause_processing=False,
            threads=[], task_queue=Queue(), completed_task_ids=set(), pending_task_ids=set(),
            failed_task_ids=set(), purged_task_ids=set(), task_metadata={},
            question_history=[], questions_list=[], quality_scores={}, quality_review_ids=set(),
            state_file_lock=threading.Lock(), task_queue_ui_lock=threading.Lock(),
            task_retry_lock=threading.Lock(), task_retry_counts={}, question_history_lock=threading.Lock(),
            quality_lock=threading.Lock(), quality_review_lock=threading.Lock(), stats_lock=threading.Lock(),
            loaded_api_processed_tasks_snapshot=None, master_duplication_enabled_var=Var(False),
            root=self.root, db_pool=None, num_threads=1,
        )
        for key in ('system_prompt_counter', 'character_counter', 'total_attempts_global',
                    'refusal_count_total', 'user_speaking_count_total', 'slop_count_total',
                    'error_count_total', 'anti_slop_count_total', 'total_input_tokens',
                    'total_output_tokens', 'estimated_cost'):
            setattr(self.state, key, 0)
        for key in ('refusal_counts_per_api', 'user_speaking_counts_per_api', 'slop_counts_per_api',
                    'error_counts_per_api', 'total_attempts_per_api', 'anti_slop_counts_per_api'):
            setattr(self.state, key, {i: 0 for i in range(6)})
        for name in ('refusals', 'user_speaking', 'slop', 'errors', 'anti_slop'):
            setattr(self.state, 'recent_{}_per_api'.format(name), {i: [] for i in range(6)})
            setattr(self.state, 'recent_{}_total'.format(name), [])
        self.ns = {
            'os': os, 'json': json, 'tempfile': tempfile, 'deepcopy': deepcopy, 'nullcontext': nullcontext,
            'Queue': Queue, 'Empty': Empty, 'Lock': threading.Lock,
            'threading': threading, 'time': time, 'random': random,
            'DatasetRun': DatasetRun, 'has_active_sources': has_active_sources,
            'app_state': self.state,
            'global_config': SimpleNamespace(get=lambda key, default=None: self.config.get(key, default), load=lambda: None),
            'STATE_FILE_PATH': str(self.state_path), 'INPUT_DIR': str(self.directory / 'missing-input'),
            'QUESTIONS_FILE_PATH': str(self.directory / 'missing-input' / 'questions.txt'),
            'log_message': lambda *args: self.calls.append(('log', args)),
            'task_retry_counts': self.state.task_retry_counts,
            'task_retry_lock': self.state.task_retry_lock,
            'api_response_times_lock': threading.Lock(), 'api_response_times_per_slot': {i: [] for i in range(6)},
            'anti_slop_counts_per_api': self.state.anti_slop_counts_per_api,
            'loaded_processed_tasks_snapshot': None,
            'messagebox': SimpleNamespace(askyesno=Mock(return_value=True), askokcancel=Mock(return_value=True),
                                          showerror=Mock(), showinfo=Mock()),
            'toast': SimpleNamespace(show=Mock()),
            'update_dashboard': Mock(), 'update_queue_ui': Mock(), 'update_database_status': Mock(),
            'tk': SimpleNamespace(NORMAL='normal', DISABLED='disabled'),
            'ttk': SimpleNamespace(Progressbar=Widget, Label=Widget, Frame=Widget),
            'progress_frame': Widget(), 'SPACING': 8,
            'api_handler': SimpleNamespace(valkey_client=None),
            'api_profiles': SimpleNamespace(DEFAULT_PROFILE='auto', load_profiles=Mock(),
                                            reset_slot_profiles=Mock(), set_slot_profile=Mock(return_value='fixture')),
            'global_rate_limiter': SimpleNamespace(set_rate_limit=Mock()),
            'num_threads_var': Var('1'), 'no_user_impersonation_var': Var(False),
            'dashboard_pause_var': Var(False), 'worker': Mock(),
            'DEFAULT_GREETING_PROMPT': 'Greeting', 'DEFAULT_FIRST_LINE_PROMPT': 'First line',
            'DEFAULT_GREETING_CONTEXT': 'Context', 'DEFAULT_FIRST_LINE_STYLES': ('brief',),
            'OPENING_TASK_FILE': 'opening',
            'wrap_label': Mock(),
        }
        for name in ('start_button', 'pause_button', 'stop_clear_button', 'quit_button'):
            self.ns[name] = Widget()
        extracted_functions('generation.py', {
            'opening_mode_mismatch', 'opening_mode_conflict', 'save_generation_state', 'enqueue_opening_tasks',
        }, self.ns)
        extracted_functions('generate.py', {
            'GenerationStateRestoreError', '_parse_dataset_job_tracking', '_install_dataset_job_tracking',
            '_parse_saved_statistics', '_close_dataset_runtime', '_discard_dataset_checkpoint',
            '_wait_for_worker_exit', '_discard_queued_tasks', 'load_generation_state',
            'reset_all_stats_and_history', 'generation_running', '_enqueue_generation_task',
            '_queue_restored_dataset_tasks', 'start_processing', 'stop_and_clear_processing_job',
            'wait_for_threads_to_stop_for_clear', 'finalize_stop_and_clear_ui',
            '_finish_quit_after_workers', 'retry_failed_tasks', 'force_recovery',
        }, self.ns)

    def make_run(self, count=4, bind_count=1):
        runtime = DatasetRun(self.settings, str(self.state_path), run_config=RUN_CONFIG)
        self.addCleanup(runtime.close)
        tasks = [runtime.add_task(('same-original-id', 'original.txt', i, 'Original question {}'.format(i)))
                 for i in range(count)]
        runtime.seal()
        for task in tasks[:bind_count]:
            runtime.bind_task(task)
        return runtime, tasks

    def checkpoint(self, runtime, **overrides):
        data = {
            'dataset_run': runtime.snapshot(), 'config_snapshot': dict(runtime.run_config),
            'completed_task_ids': [], 'pending_task_ids': list(runtime.task_ids),
            'failed_task_ids': [], 'purged_task_ids': [],
            'task_retry_counts': {}, 'task_retry_metadata': {},
            'total_attempts_global': 3, 'processed_tasks_snapshot': 0,
        }
        data.update(overrides)
        self.state_path.write_text(json.dumps(data), encoding='utf-8')
        return data

    def assert_load_fails_without_overwrite(self, pattern):
        before = self.state_path.read_bytes()
        with self.assertRaisesRegex(self.ns['GenerationStateRestoreError'], pattern):
            self.ns['load_generation_state']()
        self.assertTrue(self.state.checkpoint_restore_blocked)
        self.assertFalse(self.ns['save_generation_state']())
        with self.assertRaises(RuntimeError):
            self.ns['save_generation_state'](required=True)
        self.assertEqual(self.state_path.read_bytes(), before)

    def test_new_dataset_job_ids_keep_original_payloads_even_when_old_ids_repeat(self):
        runtime = DatasetRun(self.settings, str(self.state_path), run_config=RUN_CONFIG)
        self.addCleanup(runtime.close)
        original = ('duplicate-old-id', 'book.txt', 7, 'Subject {{ds:literal}}', 'Context\nsecond line')
        first = self.ns['_enqueue_generation_task'](original, self.state.task_queue, runtime)
        second = self.ns['_enqueue_generation_task'](original, self.state.task_queue, runtime)
        self.assertNotEqual(first[0], second[0])
        self.assertEqual(first[1:], original[1:])
        self.assertEqual(second[1:], original[1:])
        self.assertEqual([self.state.task_queue.get_nowait(), self.state.task_queue.get_nowait()], [first, second])
        self.assertEqual(self.state.pending_task_ids, {first[0], second[0]})
        self.assertEqual(self.state.task_metadata[second[0]]['task_data'], second)

    def test_legacy_queue_payload_and_id_remain_unchanged(self):
        task = ('q_3', 'questions.txt', 3, 'Question')
        self.assertEqual(self.ns['_enqueue_generation_task'](task, self.state.task_queue), task)
        self.assertEqual(self.state.task_queue.get_nowait(), task)

    def test_restore_uses_catalog_and_terminal_states_without_rereading_inputs(self):
        runtime, tasks = self.make_run()
        saved_binding = dict(runtime.bind_task(tasks[0]).values)
        self.checkpoint(runtime, completed_task_ids=[tasks[0][0]], failed_task_ids=[tasks[1][0]],
                        purged_task_ids=[tasks[3][0]], pending_task_ids=[tasks[0][0], tasks[2][0]],
                        task_retry_counts={tasks[1][0]: 3}, task_retry_metadata={tasks[1][0]: 2},
                        processed_tasks_snapshot=1, anti_slop_counts_per_api={'5': 7})
        runtime.close()
        self.csv_path.unlink()
        self.assertTrue(self.ns['load_generation_state']())
        restored = self.state.dataset_runtime
        self.addCleanup(restored.close)
        self.assertEqual(self.state.completed_task_ids, {tasks[0][0]})
        self.assertEqual(self.state.pending_task_ids, {tasks[2][0]})
        self.assertEqual(self.state.failed_task_ids, {tasks[1][0]})
        self.assertEqual(self.state.purged_task_ids, {tasks[3][0]})
        self.assertEqual(self.state.task_metadata[tasks[1][0]],
                         {'file': 'original.txt', 'retries': 2, 'task_data': tasks[1]})
        self.assertEqual(self.state.task_retry_counts, {tasks[1][0]: 3})
        self.assertEqual(self.state.anti_slop_counts_per_api[5], 7)
        self.assertEqual(dict(restored.bind_task(tasks[0]).values), saved_binding)
        queue = Queue()
        self.assertEqual(self.ns['_queue_restored_dataset_tasks'](restored, queue), 1)
        self.assertEqual(queue.get_nowait(), tasks[2])
        self.assertTrue(queue.empty())

    def test_bad_stats_leave_the_installed_run_and_checkpoint_untouched(self):
        runtime, tasks = self.make_run()
        data = self.checkpoint(runtime, completed_task_ids=[tasks[0][0]], refusal_counts_per_api={'invalid-slot': 1})
        installed = SimpleNamespace(close=Mock())
        self.state.dataset_runtime = installed
        self.state.completed_task_ids = {'installed-job'}
        self.state.total_attempts_global = 77
        self.ns['loaded_processed_tasks_snapshot'] = 42
        self.assert_load_fails_without_overwrite('refusal_counts_per_api')
        self.assertIs(self.state.dataset_runtime, installed)
        installed.close.assert_not_called()
        self.assertEqual(self.state.completed_task_ids, {'installed-job'})
        self.assertEqual(self.state.total_attempts_global, 77)
        self.assertEqual(self.ns['loaded_processed_tasks_snapshot'], 42)
        data['refusal_counts_per_api'] = {'0': 2}
        self.state_path.write_text(json.dumps(data), encoding='utf-8')
        self.assertTrue(self.ns['load_generation_state']())
        self.addCleanup(self.state.dataset_runtime.close)
        self.assertFalse(self.state.checkpoint_restore_blocked)
        installed.close.assert_called_once()

    def test_completed_job_requires_a_durable_saved_binding(self):
        runtime, tasks = self.make_run(bind_count=0)
        self.checkpoint(runtime, completed_task_ids=[tasks[0][0]])
        self.assert_load_fails_without_overwrite('no saved row binding')
        self.assertIsNone(self.state.dataset_runtime)

    def test_main_checkpoint_and_manifest_config_must_match(self):
        runtime, _ = self.make_run()
        changed = dict(RUN_CONFIG, **{'generation.num_turns': 2})
        self.config['generation.num_turns'] = 2
        self.checkpoint(runtime, config_snapshot=changed)
        self.assert_load_fails_without_overwrite('do not match')

    def test_changed_active_alias_is_a_strict_resume_error(self):
        runtime, _ = self.make_run()
        self.checkpoint(runtime)
        self.config['datasets']['sources'][0]['column_aliases']['name'] = 'different_person'
        self.assert_load_fails_without_overwrite('settings')

    def test_changed_critical_config_does_not_offer_legacy_override(self):
        runtime, _ = self.make_run()
        self.checkpoint(runtime)
        self.config['generation.num_turns'] = 2
        self.assert_load_fails_without_overwrite('Restore the saved run settings')
        self.ns['messagebox'].askyesno.assert_not_called()

    def test_unknown_saved_job_id_is_rejected_without_installing_runtime(self):
        runtime, _ = self.make_run()
        self.checkpoint(runtime, failed_task_ids=['not-in-this-run'])
        self.assert_load_fails_without_overwrite('outside its dataset run')
        self.assertIsNone(self.state.dataset_runtime)

    def test_legacy_work_cannot_be_silently_given_new_dataset_bindings(self):
        self.state_path.write_text(json.dumps({'config_snapshot': RUN_CONFIG, 'completed_task_ids': ['q_1']}), encoding='utf-8')
        self.assert_load_fails_without_overwrite('predates dataset job bindings')

    def test_empty_legacy_checkpoint_allows_a_first_dataset_run(self):
        self.state_path.write_text(json.dumps({'config_snapshot': RUN_CONFIG, 'completed_task_ids': []}), encoding='utf-8')
        self.assertTrue(self.ns['load_generation_state']())
        self.assertIsNone(self.state.dataset_runtime)

    def test_reset_refuses_to_replace_state_while_any_worker_is_alive(self):
        runtime = SimpleNamespace(close=Mock())
        alive = SimpleNamespace(is_alive=lambda: True)
        self.state.dataset_runtime = runtime
        self.state.threads = [alive]
        self.state.pending_task_ids = {'old'}
        with self.assertRaisesRegex(RuntimeError, 'workers are still alive'):
            self.ns['reset_all_stats_and_history']()
        self.assertEqual(self.state.pending_task_ids, {'old'})
        runtime.close.assert_not_called()
        self.state.threads = []
        self.state.checkpoint_restore_blocked = True
        self.state.task_retry_counts['old'] = 5
        self.ns['reset_all_stats_and_history']()
        runtime.close.assert_called_once()
        self.assertIsNone(self.state.dataset_runtime)
        self.assertFalse(self.state.pending_task_ids)
        self.assertFalse(self.state.task_retry_counts)
        self.assertFalse(self.state.checkpoint_restore_blocked)

    def start_waiting_worker(self):
        release = threading.Event()
        worker = JoinObservedThread(release)
        worker.start()
        self.addCleanup(lambda: (release.set(), worker.join(2)))
        return release, worker

    def test_clear_waits_for_real_worker_exit_before_discarding_bound_state(self):
        runtime, tasks = self.make_run()
        self.checkpoint(runtime)
        self.state.dataset_runtime = runtime
        self.state.processing_active = True
        self.state.clearing_job = True
        self.state.pending_task_ids = {task[0] for task in tasks}
        old_queue = self.state.task_queue
        for task in tasks:
            old_queue.put(task)
        release, worker = self.start_waiting_worker()
        old_threads = [worker]
        self.state.threads = old_threads
        errors = []

        def clear():
            try:
                self.ns['wait_for_threads_to_stop_for_clear'](tuple(old_threads), old_queue, runtime)
            except Exception as exc:
                errors.append(exc)

        waiter = threading.Thread(target=clear, daemon=True)
        waiter.start()
        self.assertTrue(worker.join_attempted.wait(2))
        self.assertIs(self.state.task_queue, old_queue)
        self.assertIs(self.state.threads, old_threads)
        self.assertIs(self.state.dataset_runtime, runtime)
        self.assertTrue(self.ns['generation_running']())
        self.assertTrue(self.state_path.exists())
        release.set()
        waiter.join(3)
        self.assertFalse(waiter.is_alive())
        self.assertFalse(errors)
        self.assertIsNot(self.state.task_queue, old_queue)
        self.assertEqual(old_queue.unfinished_tasks, 0)
        self.assertEqual(self.state.threads, [])
        self.assertIsNone(self.state.dataset_runtime)
        self.assertFalse(self.state.pending_task_ids)
        self.assertFalse(self.state_path.exists())
        self.assertFalse((self.directory / 'dataset_runs' / runtime.run_id).exists())
        self.assertTrue(self.state.clearing_job)  # Start remains blocked until main-thread UI finalization.
        self.ns['finalize_stop_and_clear_ui']()
        self.assertFalse(self.ns['generation_running']())

    def test_late_clear_callback_cannot_replace_another_runs_queue(self):
        old_queue = Queue()
        runtime = SimpleNamespace(close=Mock(), discard_checkpoint=Mock())
        new_queue = self.state.task_queue
        self.state.pending_task_ids = {'new-job'}
        self.ns['wait_for_threads_to_stop_for_clear']((), old_queue, runtime)
        self.assertIs(self.state.task_queue, new_queue)
        self.assertEqual(self.state.pending_task_ids, {'new-job'})
        runtime.close.assert_not_called()
        runtime.discard_checkpoint.assert_not_called()

    def prepare_failed_job(self, live_worker):
        runtime, tasks = self.make_run(count=1)
        task = tasks[0]
        self.state.dataset_runtime = runtime
        self.state.failed_task_ids = {task[0]}
        self.state.task_metadata = {task[0]: {'file': task[1], 'retries': 4, 'task_data': task}}
        self.state.task_retry_counts[task[0]] = 4
        self.state.threads = [SimpleNamespace(is_alive=lambda: live_worker)]
        return runtime, task

    def test_manual_retry_resets_both_retry_counts_and_keeps_original_binding(self):
        runtime, task = self.prepare_failed_job(True)
        binding = runtime.bind_task(task)
        self.ns['retry_failed_tasks']()
        self.assertEqual(self.state.task_queue.get_nowait(), task)
        self.assertEqual(self.state.pending_task_ids, {task[0]})
        self.assertFalse(self.state.failed_task_ids)
        self.assertNotIn(task[0], self.state.task_retry_counts)
        self.assertEqual(self.state.task_metadata[task[0]]['retries'], 0)
        self.assertIs(runtime.bind_task(task), binding)
        self.assertEqual(len(runtime.tasks), 1)

    def test_idle_retry_is_durable_and_requeues_original_job_on_next_start(self):
        runtime, task = self.prepare_failed_job(False)
        self.ns['retry_failed_tasks']()
        self.assertTrue(self.state.task_queue.empty())
        runtime.close()
        self.assertTrue(self.ns['load_generation_state']())
        restored = self.state.dataset_runtime
        self.addCleanup(restored.close)
        queue = Queue()
        self.assertEqual(self.ns['_queue_restored_dataset_tasks'](restored, queue), 1)
        self.assertEqual(queue.get_nowait(), task)
        self.assertEqual(dict(restored.bind_task(task).values), {'person': 'Ada', 'setting': 'Library'})

    def test_missing_retry_payload_remains_failed(self):
        self.state.failed_task_ids = {'missing'}
        self.state.task_metadata = {'missing': {'file': 'old.txt', 'retries': 4}}
        self.ns['retry_failed_tasks']()
        self.assertEqual(self.state.failed_task_ids, {'missing'})
        self.assertFalse(self.state.pending_task_ids)
        self.assertTrue(self.state.task_queue.empty())

    def test_force_recovery_does_not_reset_just_restored_progress(self):
        runtime, tasks = self.make_run()
        self.checkpoint(runtime, completed_task_ids=[tasks[0][0]])
        self.ns['reset_all_stats_and_history'] = Mock(side_effect=AssertionError('Recovery must retain loaded state'))
        self.ns['force_recovery']()
        self.addCleanup(self.state.dataset_runtime.close)
        self.assertEqual(self.state.completed_task_ids, {tasks[0][0]})
        self.ns['update_dashboard'].assert_called_once()
        self.ns['update_queue_ui'].assert_called_once()

    def test_shutdown_waits_before_save_close_and_ui_destruction(self):
        release, worker = self.start_waiting_worker()
        runtime = SimpleNamespace(close=Mock())
        self.ns['save_generation_state'] = Mock()
        self.state.processing_active = True
        self.state.shutting_down = True
        waiter = threading.Thread(target=self.ns['_finish_quit_after_workers'],
                                  args=((worker,), self.state.task_queue, runtime), daemon=True)
        waiter.start()
        self.assertTrue(worker.join_attempted.wait(2))
        self.ns['save_generation_state'].assert_not_called()
        runtime.close.assert_not_called()
        self.root.destroy.assert_not_called()
        release.set()
        waiter.join(3)
        self.assertFalse(waiter.is_alive())
        self.ns['save_generation_state'].assert_called_once()
        runtime.close.assert_called_once()
        self.assertFalse(self.state.processing_active)
        callback = self.root.after.call_args.args[1]
        callback()
        self.root.destroy.assert_called_once()

    def prepare_captured_start(self, runtime=None):
        if runtime is not None:
            self.checkpoint(runtime)
        captured = []
        events = []

        class CapturedThread:
            def __init__(self, target=None, args=(), kwargs=None, name=None):
                self.target, self.args, self.kwargs, self.name = target, args, kwargs or {}, name
                captured.append(self)

            def start(self):
                events.append(('thread-start', self.name))

            def is_alive(self):
                return False

        self.ns['threading'] = SimpleNamespace(Thread=CapturedThread)
        self.ns['os'] = SimpleNamespace(path=os.path, getenv=lambda key, default=None: default,
                                       listdir=Mock(side_effect=AssertionError('Resume must not scan input files')),
                                       remove=os.remove, makedirs=os.makedirs, replace=os.replace,
                                       fsync=os.fsync, unlink=os.unlink)
        self.ns['read_txt'] = Mock(side_effect=AssertionError('Resume must not reread questions'))
        self.ns['read_book'] = Mock(side_effect=AssertionError('Resume must not reread books'))
        real_save = self.ns['save_generation_state']

        def save(required=False, expected_queue=None):
            events.append(('save', required))
            return real_save(required=required, expected_queue=expected_queue)

        self.ns['save_generation_state'] = save
        return captured, events

    def prepare_fresh_start(self, files=()):
        captured, events = self.prepare_captured_start()
        self.ns['cleanup_old_files_and_backup_output'] = Mock()
        self.ns['os'].listdir = Mock(return_value=list(files))
        return captured, events

    def test_fresh_questions_register_all_original_payloads_before_source_reads(self):
        captured, events = self.prepare_fresh_start()
        self.ns['read_txt'] = Mock(return_value=['First original question', 'Second {{ds:literal}} question'])
        self.csv_path.unlink()  # Initialization/sealing must not need a dataset row.
        self.ns['start_processing']()
        runtime = self.state.dataset_runtime
        self.addCleanup(runtime.close)
        self.assertEqual([task[1:] for task in runtime.tasks], [
            ('questions.txt', 0, 'First original question'),
            ('questions.txt', 1, 'Second {{ds:literal}} question'),
        ])
        self.assertEqual(len(runtime.task_ids), 2)
        self.assertEqual(runtime.bound_count, 0)
        self.assertEqual(tuple(self.state.task_queue.queue), runtime.tasks)
        self.assertEqual(events[0], ('save', True))
        self.assertTrue(any(thread.name == 'Worker-0' for thread in captured))

    def test_fresh_random_chunks_get_distinct_jobs_when_same_chunk_is_sampled_twice(self):
        self.prepare_fresh_start(['book.txt'])
        self.config.update({'prompts.use_questions_file': False, 'generation.num_random_chunks': 2,
                            'generation.subject_size': 10, 'generation.context_size': 20})
        self.ns['read_book'] = Mock(return_value='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' * 3)
        self.ns['random'] = SimpleNamespace(choice=lambda choices: choices[0], randint=lambda first, last: 5)
        self.ns['start_processing']()
        runtime = self.state.dataset_runtime
        self.addCleanup(runtime.close)
        self.assertEqual(len(runtime.task_ids), 2)
        self.assertEqual([task[1:] for task in runtime.tasks], [
            ('book.txt', 5, '56789ABCDE', '0123456789ABCDEFGHIJ'),
            ('book.txt', 5, '56789ABCDE', '0123456789ABCDEFGHIJ'),
        ])
        self.assertEqual(tuple(self.state.task_queue.queue), runtime.tasks)

    def test_fresh_bookless_opening_jobs_are_also_registered_with_runtime(self):
        self.prepare_fresh_start()
        self.config.update({'prompts.use_questions_file': False, 'prompts.opening_mode': True,
                            'generation.num_random_chunks': 2})
        self.ns['start_processing']()
        runtime = self.state.dataset_runtime
        self.addCleanup(runtime.close)
        self.assertEqual([task[1:] for task in runtime.tasks], [('opening', 0, '', ''), ('opening', 1, '', '')])
        self.assertEqual(tuple(self.state.task_queue.queue), runtime.tasks)
        self.assertEqual(self.state.pending_task_ids, runtime.task_ids)

    def test_idle_clear_invalidates_old_completion_callback_and_checkpoint(self):
        runtime, _ = self.make_run(count=1)
        captured, events = self.prepare_captured_start(runtime)
        self.ns['start_processing']()
        active_runtime = self.state.dataset_runtime
        self.addCleanup(active_runtime.close)
        completion = next(thread for thread in captured if thread.name == 'CompletionWaiter')
        old_queue = self.state.task_queue
        self.state.processing_active = False
        self.ns['stop_and_clear_processing_job']()
        self.assertIsNot(self.state.task_queue, old_queue)
        self.assertFalse(self.state_path.exists())
        before = list(events)
        completion.target()
        self.assertEqual(events, before)
        self.assertFalse(self.state_path.exists())
        self.assertIsNone(self.state.dataset_runtime)

    def test_start_restores_original_jobs_and_saves_before_any_workers_start(self):
        runtime, tasks = self.make_run(count=2, bind_count=1)
        captured, events = self.prepare_captured_start(runtime)
        runtime.close()
        self.csv_path.unlink()
        self.ns['start_processing']()
        restored = self.state.dataset_runtime
        self.addCleanup(restored.close)
        workers = [thread for thread in captured if thread.name.startswith('Worker-')]
        self.assertEqual(len(workers), 1)
        self.assertIs(workers[0].kwargs['dataset_runtime_local'], restored)
        self.assertEqual(tuple(self.state.task_queue.queue), tuple(tasks))
        self.assertEqual(events[0], ('save', True))
        self.assertTrue(self.state.processing_active)
        self.ns['read_txt'].assert_not_called()
        self.ns['read_book'].assert_not_called()
        self.ns['os'].listdir.assert_not_called()
        saved = json.loads(self.state_path.read_text(encoding='utf-8'))
        self.assertEqual(saved['dataset_run']['run_id'], runtime.run_id)

    def test_required_startup_checkpoint_failure_prevents_worker_start(self):
        runtime, _ = self.make_run(count=1)
        captured, _ = self.prepare_captured_start(runtime)
        self.ns['save_generation_state'] = Mock(side_effect=OSError('fixture disk failure'))
        self.ns['start_processing']()
        self.addCleanup(self.state.dataset_runtime.close)
        self.assertEqual(captured, [])
        self.assertFalse(self.state.processing_active)
        self.assertEqual(self.ns['start_button'].values['state'], 'normal')
        self.ns['save_generation_state'].assert_called_once_with(required=True)


if __name__ == '__main__':
    unittest.main()
