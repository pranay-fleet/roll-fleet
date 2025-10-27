# How to Use ClickAndReadEnv with Tasks Parameter

## Change Summary

The `ClickAndReadEnv` now accepts `tasks` as a constructor parameter instead of loading them internally. This allows you to:
1. Load tasks once and reuse them across multiple env instances
2. Control which tasks are used
3. Avoid repeated Fleet API calls

## Old Way (Before):
```python
# Each env instance loads tasks from Fleet
env = gem.make("click_and_read", max_steps=10)
# Internally calls: self.tasks = fleet.load_tasks(env_key="fira")
```

## New Way (After):
```python
import fleet
import gem

# Load tasks once
tasks = fleet.load_tasks(env_key="fira")

# Pass tasks to each env instance
env = gem.make("click_and_read", tasks=tasks, max_steps=10)
```

## Usage in Training Pipeline

The environment manager will need to load tasks and pass them when creating environments:

```python
# In your pipeline code or config:
import fleet

# Load tasks once at the start
fleet_tasks = fleet.load_tasks(env_key="fira")

# When creating environments:
env = gem.make(
    env_id="click_and_read",
    tasks=fleet_tasks,  # Pass the loaded tasks
    max_steps=30,
    format_penalty=-0.1,
    render_mode="rgb_array"
)
```

## Benefits

1. **Performance**: Tasks loaded once instead of per-env
2. **Flexibility**: Can pass different task sets to different envs
3. **Control**: Can filter/select specific tasks before passing
4. **Efficiency**: Reduces Fleet API calls

## Example: Filtered Tasks

```python
import fleet

# Load all tasks
all_tasks = fleet.load_tasks(env_key="fira")

# Filter to specific task types
jira_tasks = [t for t in all_tasks if "jira" in t.key.lower()]

# Create env with filtered tasks
env = gem.make("click_and_read", tasks=jira_tasks, max_steps=10)
```

## Configuration Update

Update your environment config to specify Fleet env_key:

```yaml
# In config file
fleet_env_key: "fira"  # Which Fleet environment to load tasks from

custom_env:
  ClickAndRead:
    env_type: click_and_read
    env_config:
      # tasks parameter will be passed programmatically
      max_steps: 30
```

## Constructor Signature

```python
def __init__(
    self,
    tasks,  # REQUIRED: List of Fleet tasks
    max_steps=5,
    image_size=100,  # Not used (kept for compatibility)
    button_size=25,  # Not used (kept for compatibility)
    format_penalty=-0.01,
    render_mode="rgb_array",
    **kwargs
):
```

The `tasks` parameter is now **required** - you must pass it when creating the environment.

