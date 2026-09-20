import os

path = r"C:\Users\susan\Downloads\intain-loan-intelligence-dashboard\components\loan-intelligence-dashboard.tsx"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

target1 = "async function run() { if (!loanId.trim()) return; setLoading(true); setError(''); setSelected(null); try { const data = await runPipeline(loanId.trim()); setResult(data); setStages(normalizeStages(data)) } catch (e) { setError(errorText(e)); setStages([]) } finally { setLoading(false) } }"

repl1 = """async function run() {
    if (!loanId.trim()) return;
    setLoading(true);
    setError('');
    setSelected(null);
    setResult(null);

    let currentStages = stageOrder.map(key => ({ key, label: stageLabels[key], status: 'waiting' as StageStatus }));
    setStages([...currentStages]);

    let cumulativeResult: PipelineResponse = { loan_id: loanId.trim(), stages: [] };

    try {
      for (let i = 0; i < stageOrder.length; i++) {
        const key = stageOrder[i];
        
        currentStages[i] = { ...currentStages[i], status: 'running' };
        setStages([...currentStages]);

        const data = await retryStage(loanId.trim(), key);
        const stageData = data.stages?.[0];

        if (stageData) {
           currentStages[i] = stageData;
           cumulativeResult.stages.push(stageData);
           setResult({ ...data, stages: cumulativeResult.stages });
        } else {
           currentStages[i] = { ...currentStages[i], status: 'completed' };
        }

        setStages([...currentStages]);

        if (currentStages[i].status === 'failed') break;
        
        await new Promise(r => setTimeout(r, 400));
      }
    } catch (e) {
      setError(errorText(e));
    } finally {
      setLoading(false);
    }
  }"""

if target1 in content:
    content = content.replace(target1, repl1)
    print("Replaced run function")
else:
    print("Could not find run function target")

target2 = "valueAt(prediction, 'next_state', 'predicted_state')"
repl2 = "(prediction?.next_state as any)?.predicted_state"

if target2 in content:
    content = content.replace(target2, repl2)
    print("Replaced json formatting")
else:
    print("Could not find json formatting target")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("Done")
