# Prompt Flow for Click and Read Environment

## Overview
This document explains how the task-specific prompt (`task.prompt`) from Fleet flows through the system to reach the LLM at runtime.

## Flow Diagram

```
Fleet Task (runtime)
    ↓
task.prompt = "Create a new Jira ticket for bug XYZ"
    ↓
env.reset() → selected_task = random.choice(tasks)
    ↓
get_instructions() → formats task.prompt + tool descriptions
    ↓
Returns as env_instruction in info dict
    ↓
VLTrajEnvManager.format_messages()
    ↓
At step 0: prepends env_instruction to pre_step_template
    ↓
LLM receives full prompt
```

## Detailed Prompt Construction

### Step 1: Reset Environment
When `reset()` is called:
```python
self.selected_task = random.choice(self.tasks)  # Random Fleet task
observation, info = env.reset(seed=42)
# info = {"env_instruction": get_instructions()}
```

### Step 2: Generate Instructions
`get_instructions()` dynamically includes the task:
```python
def get_instructions(self) -> str:
    task_prompt = self.selected_task.prompt  # e.g., "Create a Jira ticket..."
    return f"""TASK: {task_prompt}
    
    You are viewing a web browser screenshot...
    Available Tools: ...
    """
```

### Step 3: VLTrajEnvManager Formats Messages
At step 0, the framework automatically prepends `env_instruction`:

```python
# From vl_traj_env_manager.py line 154-155
pre_step_content = self.pre_step_template.format(turn_idx=idx + 1)
if self.rollout_cache.step == 0:
    pre_step_content = history.history[0]["env_instruction"] + pre_step_content
```

### Step 4: Final LLM Input
The LLM receives this structure:

```
SYSTEM MESSAGE:
    You are an AI assistant that can interact with web browsers to solve tasks.
    
    You have access to the following tools:
    1. browser - Interact with the web page:
       - browser(action="left_click", x=X, y=Y)
       ...
    2. complete_task - Signal when done
    3. give_up - Last resort
    
    Important guidelines: ...

USER MESSAGE (Step 0):
    TASK: Create a new Jira ticket for bug XYZ in project CORE
    
    You are viewing a web browser screenshot...
    Available Tools: ...
    
    --- Turn 1 ---
    Screenshot:
    [IMAGE: Browser screenshot showing Jira interface]
    
    Actions remaining: 10
    Max response length: 128 tokens
    
    Decide your next action (use one tool):

ASSISTANT:
    browser(action="left_click", x=250, y=150)

USER MESSAGE (Step 1):
    --- Turn 2 ---
    Screenshot:
    [IMAGE: Updated screenshot after click]
    
    Actions remaining: 9
    ...
```

## Key Configuration Points

### In `traj_envs_click_and_read.yaml`:

1. **agent_system_template** - Sets the agent's role and tool descriptions
   - Shown once at the start
   - Provides global context and capabilities

2. **pre_step_template** - Shown before each screenshot
   - At step 0: `env_instruction` (with task.prompt) is automatically prepended
   - At step 1+: Just the turn marker

3. **next_step_template** - Shown after each screenshot
   - Reminds agent about actions left
   - Prompts for next action

### In `env.py`:

```python
def get_instructions(self):
    # This is called at reset() and returned as env_instruction
    # It includes the dynamic task.prompt from Fleet
    return f"""TASK: {self.selected_task.prompt}
    
    Available Tools: ...
    Examples: ...
    """
```

## Why This Design?

1. **Dynamic Task Injection**: Each episode gets a different Fleet task
2. **Clean Separation**: 
   - System prompt = capabilities and behavior
   - env_instruction = specific task + tools
   - pre/next templates = conversation structure
3. **No Hardcoding**: Task descriptions come from Fleet, not config
4. **Maintainable**: Change tool descriptions in one place

## Runtime Example

```python
# Episode starts
fleet_task = random.choice(tasks)
fleet_task.prompt = "Navigate to Settings and enable 2FA"

# Reset environment
obs, info = env.reset(seed=42)
# info["env_instruction"] = "TASK: Navigate to Settings and enable 2FA\n\nAvailable Tools:..."

# VLTrajEnvManager formats this into LLM input
# Step 0: env_instruction is shown
# Step 1+: Only turn markers and screenshots
```

## Updating the Prompts

### To change tool descriptions:
Edit `agent_system_template` in config or `get_instructions()` in env.py

### To change conversation format:
Edit `pre_step_template` and `next_step_template` in config

### To change task injection:
Modify `get_instructions()` method in env.py

### To add task-specific hints:
Extend `get_instructions()` to check `task.tags` or `task.metadata`

## Testing the Prompt

You can test the full prompt by running:
```python
from roll.pipeline.agentic.env.click_and_read.env import ClickAndReadEnv
env = ClickAndReadEnv()
obs, info = env.reset()
print(info["env_instruction"])
```

This will show exactly what the LLM sees at step 0.

