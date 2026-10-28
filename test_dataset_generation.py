"""Dataset integration through the real generation worker and request builders.

Run separately: ``python -B test_dataset_generation.py``. The existing opening
test harness supplies its isolated temporary config, Tk-variable stand-ins, and
worker defaults. These tests replace only HTTP/caching/timing boundaries; real
sources, durable DatasetRun allocation, prompt builders, worker loops, fixers,
state saving, and output writing execute. No display or paid model is used.
"""

import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import tempfile
import threading
import unittest
from unittest.mock import patch

import test_opening_mode as harness
from dataset_runtime import DatasetRun
from dataset_settings import apply_schema, new_source
from dataset_sources import SourceSpec, open_source


g = harness.g
app_state = harness.app_state
CELL_LITERAL = "Cell {context} {{ds:unmapped_inside_cell}} <tag> \\path\nsecond line"
MODEL_LITERAL = "{{ds:unmapped_inside_model}} {last_assistant_message}"
RAW_LITERAL = "Source {{ds:unmapped_inside_source}} {context}"


class RecordingHTTP:
    """Thread-safe fake HTTP server; response routing depends on actual payloads."""

    def __init__(self, question_barrier=None, rewrite=False, requeue_first=False, answer_script=None):
        self.lock = threading.Lock()
        self.calls = []
        self.counts = {}
        self.question_rows = set()
        self.question_barrier = question_barrier
        self.rewrite = rewrite
        self.requeue_first = requeue_first
        self.answer_script = answer_script
        self.before_question = None
        self.active_questions = 0
        self.peak_questions = 0

    @staticmethod
    def kind(messages, url):
        if url.endswith('/slop'):
            return 'slop'
        if url.endswith('/anti'):
            return 'anti'
        if messages[0]['content'].startswith('You are a strict but fair quality evaluator'):
            return 'quality'
        return harness.Router.kind_of(messages)

    def post(self, url, headers=None, data=None, timeout=None):
        payload = json.loads(data)
        messages = payload['messages']
        kind = self.kind(messages, url)
        match = re.search(r'ROW=(R\d+)', '\n'.join(message['content'] for message in messages))
        row = match.group(1) if match else 'legacy'
        with self.lock:
            self.calls.append((kind, row, url, payload))
            key = kind, row, url
            count = self.counts.get(key, 0)
            self.counts[key] = count + 1
            first_question = kind == 'question' and row not in self.question_rows
            if first_question:
                self.question_rows.add(row)
                self.active_questions += 1
                self.peak_questions = max(self.peak_questions, self.active_questions)
        if kind == 'question':
            try:
                if first_question and self.question_barrier is not None:
                    self.question_barrier.wait(timeout=12)
                if self.before_question is not None:
                    self.before_question(row)
            finally:
                if first_question:
                    with self.lock:
                        self.active_questions -= 1
            text = 'Question ROW={} with literal {}.'.format(row, MODEL_LITERAL)
        elif kind == 'answer':
            if self.requeue_first and row == 'R000' and count == 0:
                return harness.Resp(503, 'Temporary fixture outage', {})
            if self.answer_script is not None:
                response = self.answer_script(row, count, url, messages)
                if isinstance(response, tuple):
                    return harness.Resp(response[0], response[1], {})
                text = response
            else:
                phrase = 'shivers down' if self.rewrite else 'steady movement'
                text = 'ROW={} A complete sentence with {} and literal {}.'.format(row, phrase, MODEL_LITERAL)
        elif kind in ('slop', 'anti'):
            original = messages[-1]['content'].split('<original_text>\n', 1)[1].rsplit('\n</original_text>', 1)[0]
            # Keep the problem for two iterations, so a real rotating authored
            # instruction is supplied on a later fixer call. Every result differs.
            if count % 3 < 2:
                text = original + ' Revision {}.'.format(count)
            else:
                text = original.replace('shivers down', 'steady movement').replace('secretbad', 'clear wording')
        elif kind == 'continuation':
            text = 'Continuation ROW={} retains literal {}.'.format(row, MODEL_LITERAL)
        elif kind == 'greeting':
            text = 'Greeting ROW={} with enough context and literal {}.'.format(row, MODEL_LITERAL)
        elif kind == 'first_line':
            text = 'First reply ROW={} with literal {}.'.format(row, MODEL_LITERAL)
        elif kind == 'quality':
            text = json.dumps({name: 85 for name in ('coherence', 'naturalness', 'engagement', 'consistency')})
        else:
            raise AssertionError('Unrouted request: {}'.format(kind))
        return harness.Resp(200, text, {'prompt_tokens': 11, 'completion_tokens': 7})


class DatasetGenerationTests(unittest.TestCase):
    def setUp(self):
        harness.reset_state()
        app_state.purged_task_ids.clear()
        app_state.quality_scores.clear()
        app_state.quality_review_ids.clear()
        app_state.quality_enabled = False
        app_state.quality_output_filter = False
        app_state.use_conversation_history_for_fixers = True
        app_state.master_duplication_enabled_var = harness.Var(False)
        app_state.dataset_runtime = None
        app_state.checkpoint_restore_blocked = False
        app_state.clearing_job = False
        app_state.task_queue = None
        g.MAX_TASK_REQUEUES = 2
        harness.CONFIG['generation.max_slop_sentence_fix_iterations'] = 5
        harness.CONFIG['generation.max_anti_slop_fix_iterations'] = 5
        harness.CONFIG['generation.discard_refused_conversations'] = False
        self.temporary = tempfile.TemporaryDirectory(prefix='sdg-production-datasets-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.state_path = self.directory / 'generation_state.json'
        state_patch = patch.object(g, 'STATE_FILE_PATH', str(self.state_path))
        state_patch.start()
        self.addCleanup(state_patch.stop)
        self.runtimes = []
        self.addCleanup(self._close_runtimes)
        for path in Path('output').glob('output_api_slot_*.jsonl'):
            path.unlink()
        self.preview = []
        self.preview_lock = threading.Lock()
        preview_patch = patch.object(g, 'update_live_prompt_preview', self._preview)
        preview_patch.start()
        self.addCleanup(preview_patch.stop)

    def _preview(self, messages, metadata=None):
        with self.preview_lock:
            self.preview.append(copy.deepcopy(messages))

    def _close_runtimes(self):
        for runtime in self.runtimes:
            runtime.close()
        app_state.dataset_runtime = None
        app_state.task_queue = None
        app_state.stop_processing = False

    def runtime(self, rows, count=None, turns=2, duplication=False, opening=False, questions=False):
        for collection in (app_state.completed_task_ids, app_state.failed_task_ids, app_state.pending_task_ids,
                           app_state.purged_task_ids, app_state.task_metadata, app_state.task_retry_counts):
            collection.clear()
        index = len(self.runtimes)
        people = self.directory / ('people-{}.csv'.format(index))
        with people.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['row', 'name', 'literal', 'mood'])
            for row in range(rows):
                writer.writerow(['R{:03d}'.format(row), 'Name{:03d}'.format(row), CELL_LITERAL, 'calm -> wary'])
        settings = {'sources': []}
        for kind, path, aliases in [('csv', people, {'row': 'row', 'name': 'name', 'literal': 'literal', 'mood': 'mood'})]:
            source = new_source(settings['sources'])
            source.update(kind=kind, location=str(path), column_aliases=aliases)
            reader = open_source(SourceSpec(source['source_id'], kind, str(path)))
            try:
                settings['sources'].append(apply_schema(source, reader.schema()))
            finally:
                reader.close()
        settings_path = self.directory / ('settings-{}.jsonl'.format(index))
        settings_path.write_text(''.join(json.dumps({'setting': 'Setting{:03d}'.format(row)}) + '\n'
                                         for row in range(rows + 1)), encoding='utf-8')
        source = new_source(settings['sources'])
        source.update(kind='jsonl', location=str(settings_path), column_aliases={'setting': 'setting'})
        reader = open_source(SourceSpec(source['source_id'], 'jsonl', str(settings_path)))
        try:
            settings['sources'].append(apply_schema(source, reader.schema()))
        finally:
            reader.close()
        frozen_config = {
            'prompts.use_questions_file': questions, 'generation.num_turns': turns,
            'generation.subject_size': 1000, 'generation.context_size': 3000,
            'api.master_duplication_mode': duplication, 'prompts.opening_mode': opening,
        }
        runtime = DatasetRun(settings, self.state_path, run_config=frozen_config)
        tasks = []
        for number in range(rows if count is None else count):
            original = ('original-{}'.format(number), 'fixture.txt', number,
                        RAW_LITERAL if questions else 'SUBJECT {context}', RAW_LITERAL)
            tasks.append(runtime.add_task(original[:4] if questions else original))
        runtime.seal()
        self.runtimes.append(runtime)
        app_state.dataset_runtime = runtime
        return runtime, tasks

    def kwargs(self, runtime, **overrides):
        args = harness.worker_kwargs(
            dataset_runtime_local=runtime, enable_character_engine_local=False,
            current_num_turns=2, current_max_attempts=1,
            current_system_prompts_for_worker=['SYSTEM ROW={{ds:row}} NAME={{ds:name}} CELL={{ds:literal}}'],
            current_top_level_system_prompt='TOP ROW={{ds:row}}', current_lore='LORE {{ds:setting}}',
            current_question_prompt='QUESTION-PROMPT ROW={{ds:row}} NAME={{ds:name}} SUBJECT={subject} CONTEXT={context}',
            current_answer_prompt='ANSWER-PROMPT ROW={{ds:row}} NAME={{ds:name}} CELL={{ds:literal}}',
            current_user_continuation_prompt='CONTINUE-PROMPT ROW={{ds:row}} LAST={last_assistant_message}',
            current_jailbreaks=['REFUSAL-FIX ROW={{ds:row}} NAME={{ds:name}} CELL={{ds:literal}}'],
            current_speaking_fixes=['SPEAKING-FIX ROW={{ds:row}} NAME={{ds:name}}'],
            current_slop_fixes_fallback=['FALLBACK-FIX ROW={{ds:row}} NAME={{ds:name}}'],
            current_slop_fixes_for_rotation_worker=['ROTATING-FIX ROW={{ds:row}} NAME={{ds:name}} CELL={{ds:literal}}'],
        )
        args.update(overrides)
        return args

    def queue(self, tasks, turns=2):
        q = harness.Queue()
        q.processed_tasks_lock = threading.Lock()
        q.processed_tasks = 0
        q.total_tasks_for_progress = len(tasks) * turns
        q.api_processed_tasks = {}
        q.start_times_list = []
        q.overall_task_times_list = []
        q.api_start_times_list = {}
        q.all_tasks_queued = True
        for task in tasks:
            app_state.pending_task_ids.add(task[0])
            app_state.task_metadata[task[0]] = {'file': task[1], 'retries': 0, 'task_data': task}
            q.put(task)
        app_state.task_queue = q
        return q

    def launch(self, q, kwargs, count):
        output_lock = threading.Lock()
        threads = [threading.Thread(target=g.worker, args=(number, q, output_lock), kwargs=kwargs,
                                    daemon=True) for number in range(count)]
        for thread in threads:
            thread.start()
        return threads

    def finish(self, threads, q):
        for thread in threads:
            thread.join(timeout=20)
        alive = [thread.name for thread in threads if thread.is_alive()]
        if alive:
            app_state.stop_processing = True
            for thread in threads:
                thread.join(timeout=2)
        self.assertEqual(alive, [], 'Workers did not finish: {}'.format(alive))
        self.assertEqual(q.unfinished_tasks, 0)

    def test_twenty_workers_keep_rows_through_refinement_and_automatic_requeue(self):
        runtime, tasks = self.runtime(20)
        q = self.queue(tasks)
        fixer = dict(harness.SLOT, url='http://fixture/slop')
        args = self.kwargs(runtime, slop_fixer_api_config=fixer)
        original_args = copy.deepcopy({key: value for key, value in args.items() if key != 'dataset_runtime_local'})
        server = RecordingHTTP(threading.Barrier(20), rewrite=True, requeue_first=True)
        with patch.object(g.requests, 'post', server.post):
            self.finish(self.launch(q, args, 20), q)
        self.assertEqual(server.peak_questions, 20, 'Row allocation must release its lock before model requests')
        self.assertEqual(runtime.bound_count, 20)
        self.assertEqual(app_state.completed_task_ids, {task[0] for task in tasks})
        self.assertFalse(app_state.failed_task_ids)
        self.assertFalse(app_state.pending_task_ids)
        self.assertEqual(sum(app_state.task_retry_counts.values()), 1)
        self.assertEqual(q.processed_tasks, 40)
        self.assertEqual(original_args, {key: value for key, value in args.items() if key != 'dataset_runtime_local'})
        bindings = {task[0]: runtime.bind_task(task) for task in tasks}
        values_by_row = {binding.values['row']: binding.values for binding in bindings.values()}
        self.assertEqual(len(values_by_row), 20)
        for kind, row, _, body in server.calls:
            text = '\n'.join(message['content'] for message in body['messages'])
            self.assertEqual(set(re.findall(r'ROW=(R\d+)', text)), {row})
            if kind not in ('slop', 'anti'):
                self.assertIn('NAME=' + values_by_row[row]['name'], text)
                self.assertIn(CELL_LITERAL, text)
            if kind == 'question':
                self.assertIn('SUBJECT=SUBJECT {context}', text)
                self.assertIn(RAW_LITERAL, text)
            if kind == 'continuation':
                self.assertIn(MODEL_LITERAL, body['messages'][-1]['content'])
            self.assertIn(body['messages'], self.preview)
        rotating = [body['messages'][-1]['content'] for kind, _, _, body in server.calls
                    if kind == 'slop' and 'ROTATING-FIX' in body['messages'][-1]['content']]
        self.assertGreaterEqual(len(rotating), 40)
        self.assertTrue(all(CELL_LITERAL in text and '{{ds:name}}' not in text for text in rotating))
        saved_rows = harness.read_rows()
        self.assertEqual(len(saved_rows), 20)
        for saved in saved_rows:
            expected = bindings[saved['id']].values['row']
            self.assertEqual(len(saved['conversations']), 4)
            self.assertTrue(all('ROW=' + expected in message['value'] for message in saved['conversations']))
        cache_reads = {call[1] for call in harness.CACHE_CALLS if call[0] == 'get'}
        for kind, _, _, body in server.calls:
            if kind in ('question', 'answer'):
                digest = hashlib.md5(json.dumps(body['messages'], sort_keys=True).encode()).hexdigest()
                self.assertIn((digest, 0), cache_reads)

    def test_same_worker_and_duplicate_apis_use_original_rows_and_record_completion(self):
        runtime, tasks = self.runtime(2, duplication=True)
        q = self.queue(tasks)
        app_state.master_duplication_enabled_var = harness.Var(True)
        slot_two = dict(harness.SLOT, url='http://fixture/second')
        args = self.kwargs(runtime, master_duplication_enabled_local=True,
                           all_api_configs_local=[harness.SLOT, slot_two] + [harness.EMPTY] * 4)
        server = RecordingHTTP()
        with patch.object(g.requests, 'post', server.post):
            self.finish(self.launch(q, args, 1), q)
        self.assertEqual(runtime.bound_count, 2)
        self.assertEqual(app_state.completed_task_ids, {task[0] for task in tasks})
        self.assertFalse(app_state.pending_task_ids | app_state.failed_task_ids)
        self.assertEqual(q.api_processed_tasks, {0: 4, 1: 4})
        bindings = {task[0]: runtime.bind_task(task).values['row'] for task in tasks}
        for slot in (0, 1):
            path = Path('output/output_api_slot_{}.jsonl'.format(slot))
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(records), 4)
            for record in records:
                task_id = record['id'].split('_api')[0]
                self.assertTrue(all('ROW=' + bindings[task_id] in message['value']
                                    for message in record['conversations']))
        checkpoint = json.loads(self.state_path.read_text())
        self.assertEqual(set(checkpoint['completed_task_ids']), set(bindings))
        self.assertEqual(checkpoint['dataset_run'], runtime.snapshot())

    def test_partial_duplication_stays_failed_without_automatic_replay(self):
        runtime, tasks = self.runtime(1, turns=1, duplication=True)
        q = self.queue(tasks, turns=1)
        app_state.master_duplication_enabled_var = harness.Var(True)
        slot_two = dict(harness.SLOT, url='http://fixture/second')
        server = RecordingHTTP(answer_script=lambda row, count, url, messages:
                               (503, 'Fixture outage') if url.endswith('/second') else
                               'ROW={} A complete successful response.'.format(row))
        args = self.kwargs(runtime, current_num_turns=1, master_duplication_enabled_local=True,
                           all_api_configs_local=[harness.SLOT, slot_two] + [harness.EMPTY] * 4)
        with patch.object(g.requests, 'post', server.post):
            self.finish(self.launch(q, args, 1), q)
        self.assertEqual(app_state.failed_task_ids, {tasks[0][0]})
        self.assertFalse(app_state.pending_task_ids | app_state.completed_task_ids)
        self.assertFalse(app_state.task_retry_counts)
        self.assertEqual(runtime.bound_count, 1)
        self.assertTrue(Path('output/output_api_slot_0.jsonl').exists())
        self.assertEqual(json.loads(self.state_path.read_text())['failed_task_ids'], [tasks[0][0]])

    def test_opening_templates_and_character_fragments_are_rendered_once(self):
        runtime, tasks = self.runtime(1, opening=True)
        q = self.queue(tasks)
        character = dict(harness.CHARACTERS[0], name='{{ds:name}}', setting='{{ds:setting}}',
                         backstory='PROFILE {{ds:literal}}', persona='PERSONA {{ds:name}}')
        original_character = dict(character)
        args = self.kwargs(runtime, opening_mode_local=True, enable_character_engine_local=True,
                           enable_setting_selection_local=True, character_list=[character],
                           enable_emotional_states_local=True, emotional_states_list_local=['{{ds:mood}}'],
                           greeting_prompt_local='GREETING-PROMPT ROW={{ds:row}} SPEAKER={speaker} CONTEXT={context}',
                           greeting_context_local='SETTING={{ds:setting}} CHUNK={chunk} CELL={{ds:literal}}',
                           first_line_prompt_local='FIRSTLINE-PROMPT ROW={{ds:row}} STYLE={style}',
                           first_line_styles_local=['STYLE {{ds:literal}}'])
        server = RecordingHTTP()
        with patch.object(g.requests, 'post', server.post):
            self.finish(self.launch(q, args, 1), q)
        self.assertEqual(character, original_character)
        greeting = next(body for kind, _, _, body in server.calls if kind == 'greeting')
        self.assertIn('SPEAKER=Name000', greeting['messages'][-1]['content'])
        self.assertIn(RAW_LITERAL, greeting['messages'][-1]['content'])
        self.assertIn(CELL_LITERAL, greeting['messages'][-1]['content'])
        self.assertIn('PROFILE ' + CELL_LITERAL, greeting['messages'][0]['content'])
        first_line = next(body for kind, _, _, body in server.calls if kind == 'first_line')
        self.assertIn('STYLE=STYLE ' + CELL_LITERAL, first_line['messages'][0]['content'])
        self.assertIn(MODEL_LITERAL, first_line['messages'][-1]['content'])
        answers = [body for kind, _, _, body in server.calls if kind == 'answer']
        self.assertIn('PERSONA Name000', answers[0]['messages'][0]['content'])
        self.assertIn('CALM -> WARY', answers[0]['messages'][0]['content'])
        self.assertEqual(app_state.completed_task_ids, {tasks[0][0]})

    def test_profile_greeting_and_question_file_remain_literal_after_authored_substitution(self):
        runtime, tasks = self.runtime(1, turns=1, opening=True)
        q = self.queue(tasks, turns=1)
        character = dict(harness.CHARACTERS[0], name='{{ds:name}}', greeting='ROW={{ds:row}} {{ds:literal}}')
        server = RecordingHTTP()
        args = self.kwargs(runtime, current_num_turns=1, opening_mode_local=True,
                           enable_character_engine_local=True, character_list=[character],
                           first_line_prompt_local='FIRSTLINE-PROMPT ROW={{ds:row}} {style}')
        with patch.object(g.requests, 'post', server.post):
            self.finish(self.launch(q, args, 1), q)
        self.assertFalse(any(kind == 'greeting' for kind, _, _, _ in server.calls))
        first_line = next(body for kind, _, _, body in server.calls if kind == 'first_line')
        self.assertEqual(first_line['messages'][-1]['content'], 'ROW=R000 ' + CELL_LITERAL)

        runtime, tasks = self.runtime(1, turns=1, questions=True)
        q = self.queue(tasks, turns=1)
        app_state.quality_enabled = True
        harness.CONFIG['quality.use_llm_scoring'] = True
        harness.CONFIG['quality.scoring_api'] = {'url': 'http://fixture/quality', 'model': 'fixture', 'key': ''}
        server = RecordingHTTP()
        with patch.object(g.requests, 'post', server.post):
            self.finish(self.launch(q, self.kwargs(runtime, current_num_turns=1, use_questions_file_local=True), 1), q)
        self.assertFalse(any(kind == 'question' for kind, _, _, _ in server.calls))
        answer = next(body for kind, _, _, body in server.calls if kind == 'answer')
        self.assertEqual(answer['messages'][1]['content'], RAW_LITERAL)
        quality = next(body for kind, _, _, body in server.calls if kind == 'quality')
        self.assertIn(RAW_LITERAL, quality['messages'][-1]['content'])
        self.assertIn(MODEL_LITERAL, quality['messages'][-1]['content'])

    def test_authored_refusal_speaking_fallback_and_anti_slop_fixes_use_bound_values(self):
        cases = (
            ('refusal', 'I cannot help with this', 'REFUSAL-FIX', {}),
            ('speaking', 'you nod at the result', 'SPEAKING-FIX', {}),
            ('fallback', 'shivers down with secretbad wording', 'FALLBACK-FIX', {'current_anti_slop_phrases': ['secretbad']}),
            ('anti', 'secretbad wording', 'ROTATING-FIX', {
                'current_anti_slop_phrases': ['secretbad'],
                'anti_slop_fixer_api_config_param': dict(harness.SLOT, url='http://fixture/anti'),
            }),
        )
        for name, problem, marker, options in cases:
            with self.subTest(case=name):
                runtime, tasks = self.runtime(1, turns=1)
                q = self.queue(tasks, turns=1)
                server = RecordingHTTP(answer_script=lambda row, count, url, messages, phrase=problem:
                                       'ROW={} {} and literal {}.'.format(row, phrase if count == 0 else 'A clean final response', MODEL_LITERAL))
                with patch.object(g.requests, 'post', server.post):
                    self.finish(self.launch(q, self.kwargs(runtime, current_num_turns=1, **options), 1), q)
                self.assertIn(tasks[0][0], app_state.completed_task_ids)
                amended = ['\n'.join(message['content'] for message in body['messages'])
                           for _, _, _, body in server.calls]
                self.assertTrue(any(marker + ' ROW=R000 NAME=Name000' in text for text in amended), marker)
                self.assertFalse(any('{{ds:row}}' in text or '{{ds:name}}' in text for text in amended))

    def test_invalid_later_template_fails_before_first_api_and_keeps_original_binding(self):
        for change in ({'current_answer_prompt': 'ANSWER-PROMPT {{ds:missing}}'},
                       {'current_slop_fixes_for_rotation_worker': ['bad {{ds:has-hyphen}}']}):
            with self.subTest(change=change):
                runtime, tasks = self.runtime(1)
                q = self.queue(tasks)
                server = RecordingHTTP()
                with patch.object(g.requests, 'post', server.post):
                    self.finish(self.launch(q, self.kwargs(runtime, **change), 1), q)
                self.assertEqual(server.calls, [])
                self.assertIn(tasks[0][0], app_state.failed_task_ids)
                self.assertNotIn(tasks[0][0], app_state.pending_task_ids)
                self.assertEqual(runtime.bound_count, 1)
                self.assertEqual(runtime.bind_task(tasks[0]).values['row'], 'R000')
                self.assertIn(tasks[0][0], json.loads(self.state_path.read_text())['failed_task_ids'])

    def test_source_exhaustion_leaves_inflight_bound_job_running(self):
        runtime, tasks = self.runtime(1, count=2, turns=1)
        q = self.queue(tasks, turns=1)
        server = RecordingHTTP()
        entered, release = threading.Event(), threading.Event()
        server.before_question = lambda row: (entered.set(), release.wait(10))
        args = self.kwargs(runtime, current_num_turns=1)
        with patch.object(g.requests, 'post', server.post):
            first = self.launch(q, args, 1)
            self.assertTrue(entered.wait(5))
            other = self.launch(q, args, 1)
            for _ in range(500):
                if tasks[1][0] in app_state.failed_task_ids:
                    break
                threading.Event().wait(.01)
            self.assertIn(tasks[1][0], app_state.failed_task_ids)
            self.assertTrue(runtime.failed)
            self.assertFalse(app_state.stop_processing)
            release.set()
            self.finish(first + other, q)
        self.assertIn(tasks[0][0], app_state.completed_task_ids)
        self.assertEqual(runtime.bound_count, 1)
        self.assertEqual(runtime.bind_task(tasks[0]).values['row'], 'R000')
        self.assertEqual({row for _, row, _, _ in server.calls}, {'R000'})

    def test_question_failure_exhausts_retries_without_resampling_and_saves_failed_status(self):
        runtime, tasks = self.runtime(1, turns=1)
        q = self.queue(tasks, turns=1)
        requests = []

        def unavailable(url, headers=None, data=None, timeout=None):
            requests.append(json.loads(data)['messages'])
            return harness.Resp(503, 'Fixture unavailable', {})

        with patch.object(g.requests, 'post', unavailable):
            self.finish(self.launch(q, self.kwargs(runtime, current_num_turns=1), 1), q)
        self.assertEqual(len(requests), g.MAX_TASK_REQUEUES + 1)
        self.assertTrue(all('ROW=R000' in messages[-1]['content'] for messages in requests))
        self.assertEqual(runtime.bound_count, 1)
        self.assertEqual(app_state.failed_task_ids, {tasks[0][0]})
        self.assertFalse(app_state.pending_task_ids | app_state.completed_task_ids)
        saved = json.loads(self.state_path.read_text())
        self.assertEqual(saved['failed_task_ids'], [tasks[0][0]])
        self.assertEqual(saved['pending_task_ids'], [])
        self.assertEqual(saved['task_retry_counts'][tasks[0][0]], g.MAX_TASK_REQUEUES + 1)
        self.assertEqual(saved['task_retry_metadata'][tasks[0][0]], g.MAX_TASK_REQUEUES + 1)

    def test_atomic_checkpoint_preserves_old_file_on_failure_and_uses_frozen_config(self):
        runtime, tasks = self.runtime(1)
        q = self.queue(tasks)
        runtime.bind_task(tasks[0])
        harness.CONFIG['generation.num_turns'] = 99
        self.assertTrue(g.save_generation_state(required=True))
        original = self.state_path.read_bytes()
        saved = json.loads(original)
        self.assertEqual(saved['config_snapshot']['generation.num_turns'], 2)
        self.assertEqual(saved['dataset_run'], runtime.snapshot())
        self.assertEqual(saved['pending_task_ids'], [tasks[0][0]])
        with patch.object(g.os, 'replace', side_effect=OSError('fixture replacement failure')):
            self.assertFalse(g.save_generation_state())
            with self.assertRaisesRegex(RuntimeError, 'required generation checkpoint'):
                g.save_generation_state(required=True)
        self.assertEqual(self.state_path.read_bytes(), original)
        self.assertEqual(list(self.directory.glob('.generation-state-*.tmp')), [])
        app_state.checkpoint_restore_blocked = True
        try:
            self.assertFalse(g.save_generation_state())
            with self.assertRaisesRegex(RuntimeError, 'reset explicitly'):
                g.save_generation_state(required=True)
        finally:
            app_state.checkpoint_restore_blocked = False
        self.assertEqual(self.state_path.read_bytes(), original)
        app_state.clearing_job = True
        try:
            self.assertFalse(g.save_generation_state())
            with self.assertRaisesRegex(RuntimeError, 'being cleared'):
                g.save_generation_state(required=True)
        finally:
            app_state.clearing_job = False
        self.assertEqual(self.state_path.read_bytes(), original)
        replacement_queue = harness.Queue()
        app_state.task_queue = replacement_queue
        try:
            self.assertFalse(g.save_generation_state(expected_queue=q))
            with self.assertRaisesRegex(RuntimeError, 'no longer current'):
                g.save_generation_state(required=True, expected_queue=q)
        finally:
            app_state.task_queue = q
        self.assertEqual(self.state_path.read_bytes(), original)
        self.assertEqual(q.unfinished_tasks, 1)  # Saving never consumes the queue.

    def test_no_dataset_worker_retains_legacy_literals_and_native_replacement_order(self):
        server = RecordingHTTP()
        task = harness.chunk_task(context='NATIVE')
        task = task[:3] + ('{context}', task[4])
        with patch.object(g.requests, 'post', server.post):
            harness.run_worker(task, enable_character_engine_local=False,
                               current_system_prompts_for_worker=['SYSTEM {{ds:not_enabled}}'],
                               current_question_prompt='QUESTION-PROMPT {subject}/{context}')
        question = next(body for kind, _, _, body in server.calls if kind == 'question')
        self.assertEqual(question['messages'][0]['content'], 'SYSTEM {{ds:not_enabled}}')
        self.assertEqual(question['messages'][-1]['content'], 'QUESTION-PROMPT NATIVE/NATIVE')
        self.assertEqual(app_state.completed_task_ids, {task[0]})


if __name__ == '__main__':
    unittest.main(verbosity=2)
