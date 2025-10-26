# Click and Read Environment

A simple VL (Vision-Language) + Tool Use environment for testing multi-turn RL agents.

## Task Description

1. **Initial State**: Agent sees a 100x100 image with a 25x25 green button in one of 4 corner positions (top-left, top-right, bottom-left, bottom-right)
2. **Step 1**: Agent must click the green button using coordinates `<answer>click:[x,y]</answer>`
3. **Step 2**: If clicked correctly, image changes to show a word from the vocabulary
4. **Step 3**: Agent must read and submit the word using `<answer>submit:[word]</answer>`
5. **Reward**: Terminal reward of 1.0 for correct submission (after clicking button), 0.0 otherwise

## Features

- **Simple Visual Task**: 100x100 RGB images, easy for VL models to process
- **Tool Use**: Two distinct actions (click and submit) with different formats
- **Multi-turn**: Requires 2-3 steps to complete successfully
- **Clear Reward Signal**: Binary success/failure at episode end
- **Deterministic Evaluation**: Fixed positions and word vocabulary for reproducibility
- **Fast Rollouts**: Lightweight rendering, suitable for rapid RL training

## Action Format

### Click Action
```
<answer>click:[x,y]</answer>
```
- `x`, `y`: Pixel coordinates (0-99)
- Must be within button bounds to succeed

### Submit Action
```
<answer>submit:[word]</answer>
```
- `word`: The word shown in the image
- Case-insensitive
- Must have successfully clicked button first

## Vocabulary

8 words: APPLE, BANANA, CHERRY, DRAGON, EAGLE, FOREST, GUITAR, HELLO

## Button Positions

- **Top-left**: (0, 0)
- **Top-right**: (75, 0)
- **Bottom-left**: (0, 75)
- **Bottom-right**: (75, 75)

## Configuration

Default parameters:
- `max_steps`: 5
- `image_size`: 100
- `button_size`: 25
- `format_penalty`: -0.01 (for invalid action format)

## Testing

Run the test suite:
```bash
cd roll/pipeline/agentic/env/click_and_read
python test_env.py
```

This will:
- Test successful episodes
- Test failure cases (wrong click, wrong answer, invalid format)
- Test edge cases (max steps timeout)
- Generate sample images for inspection

## Training

Use the provided configuration:
```bash
cd /workspace/ROLL  # or your ROLL directory
bash examples/qwen2.5-vl-3B-agentic/run_agentic_pipeline_click_and_read.sh
```

## Metrics

The environment tracks:
- `action_is_valid`: Whether action format was correct
- `button_clicked`: Whether button was successfully clicked
- `success`: Whether episode ended with reward > 0
- `steps_used`: Number of steps taken
- `button_position`: Which corner had the button (for analysis)

## File Structure

```
click_and_read/
├── __init__.py              # Gem registration
├── env.py                   # Main environment implementation
├── test_env.py              # Test suite
└── README.md                # This file
```

## Integration with ROLL

The environment is registered with `gem` and can be used in any ROLL pipeline:

```yaml
custom_env:
  ClickAndRead:
    env_type: click_and_read
    max_steps: 5
    env_manager_cls: roll.pipeline.agentic.env_manager.vl_traj_env_manager.VLTrajEnvManager
    # ... other config
```

