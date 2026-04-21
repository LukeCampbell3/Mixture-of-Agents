import sys
sys.path.insert(0, '.')

from app.device_profile import load_or_create
from app.models import local_llm_client as llm_mod

profile = load_or_create()
print('Worker model:', profile['models']['worker'])
print('Router model:', profile['models']['router'])

call_log = []
orig_generate = llm_mod.OllamaClient.generate

def fake_generate(self, prompt, max_tokens=600, temperature=0.7, **kw):
    call_log.append({
        'words': len(prompt.split()),
        'chars': len(prompt),
        'max_tokens': max_tokens,
        'model': self.model,
    })
    return 'Fake response for measurement.'

llm_mod.OllamaClient.generate = fake_generate

from app.orchestrator import Orchestrator

orch = Orchestrator(
    llm_provider='ollama',
    llm_model=profile['models']['worker'],
    llm_base_url='http://localhost:11434',
    router_model=profile['models']['router'],
    budget_mode=profile['runtime']['budget_mode'],
    enable_parallel=profile['runtime']['enable_parallel'],
    max_parallel_agents=profile['runtime']['max_parallel_agents'],
)

orch.run_task('how would you implement a doubly linked list in python?')

print(f'\nLLM calls made: {len(call_log)}')
for i, c in enumerate(call_log):
    est_tps = 13.0  # tok/s for 1.5b
    est_time = (c['words'] * 1.3 + c['max_tokens']) / est_tps
    print(f"  Call {i+1}: ~{c['words']} words input, max_tokens={c['max_tokens']}  => est {est_time:.0f}s @ 13 tok/s  (model: {c['model']})")

total_out = sum(c['max_tokens'] for c in call_log)
total_in  = sum(c['words'] for c in call_log)
print(f'\nTotal input words: {total_in}  Total max output tokens: {total_out}')
print(f'Estimated total time @ 13 tok/s: {(total_in*1.3 + total_out)/13:.0f}s')
