# Updated Prompt Structure with Tools in Each User Message

## Why This Change?

✅ **More explicit**: Tools visible in every turn  
✅ **Less reliance on system prompt**: Model doesn't need to remember from far back  
✅ **Better for attention**: Tools are in immediate context window  
✅ **More robust**: Works even if model ignores system prompt  

## New Structure

### System Message (Once at Step 0)
```
ROLE: system

CONTENT:
You are an AI assistant that can interact with web browsers to solve tasks.

You have access to the following tools:
[Full tool descriptions with examples]

Important guidelines:
- Analyze the screenshot carefully before taking actions
- Be methodical and break down tasks into steps
- Use coordinates within screen bounds
- Call complete_task when the task is done
```

### User Message - Turn 1 (Step 0)
```
ROLE: user

CONTENT: [
  TEXT: "TASK: Create a Jira ticket in project CORE
         
         Analyze the screenshot carefully...
         
         --- Turn 1 ---
         Screenshot:"
  
  IMAGE: <screenshot>
  
  TEXT: "
         Available tools:
         • browser(action=\"left_click\", x=X, y=Y)
         • browser(action=\"right_click\", x=X, y=Y)
         • browser(action=\"double_click\", x=X, y=Y)
         • browser(action=\"type\", text=\"...\")
         • browser(action=\"key\", text=\"Enter\")
         • browser(action=\"scroll\", x=X, y=Y, scroll_direction=\"down\", scroll_amount=3)
         • browser(action=\"wait\", duration=2)
         • browser(action=\"left_click_drag\", start_x=X1, start_y=Y1, x=X2, y=Y2)
         • complete_task(success=true/false, summary=\"...\", answer=\"...\")
         • give_up(reason=\"...\", attempts_made=[...])
         
         Actions remaining: 10
         Max response: 128 tokens
         
         Your action:"
]
```

### User Message - Turn 2+ (Step 1+)
```
ROLE: user

CONTENT: [
  TEXT: "--- Turn 2 ---
         Screenshot:"
  
  IMAGE: <screenshot>
  
  TEXT: "
         Available tools:
         • browser(action=\"left_click\", x=X, y=Y)
         • browser(action=\"right_click\", x=X, y=Y)
         • browser(action=\"double_click\", x=X, y=Y)
         • browser(action=\"type\", text=\"...\")
         • browser(action=\"key\", text=\"Enter\")
         • browser(action=\"scroll\", x=X, y=Y, scroll_direction=\"down\", scroll_amount=3)
         • browser(action=\"wait\", duration=2)
         • browser(action=\"left_click_drag\", start_x=X1, start_y=Y1, x=X2, y=Y2)
         • complete_task(success=true/false, summary=\"...\", answer=\"...\")
         • give_up(reason=\"...\", attempts_made=[...])
         
         Actions remaining: 9
         Max response: 128 tokens
         
         Your action:"
]
```

## What Appears Where Now

| Content | System | Turn 1 | Turn 2+ |
|---------|:------:|:------:|:-------:|
| **Tool descriptions (detailed)** | ✅ | ❌ | ❌ |
| **Tool list (compact)** | ❌ | ✅ | ✅ |
| **Task prompt** | ❌ | ✅ | ❌ |
| **Screenshot** | ❌ | ✅ | ✅ |
| **Action counter** | ❌ | ✅ | ✅ |

## Key Changes

### Before:
- Tools ONLY in system prompt
- User messages had no tool reminders
- Model had to remember tools from system context

### After:
- Tools in system prompt (detailed with examples)
- **Tools in EVERY user message (compact list)**
- Model sees tools right before generating action
- More robust to attention issues

## Token Impact

### Turn 1 (Step 0):
- Before: ~100 tokens (task + turn info)
- After: ~200 tokens (task + turn info + tool list)
- **+100 tokens**

### Turn 2+ (Step 1+):
- Before: ~50 tokens (turn info only)
- After: ~150 tokens (turn info + tool list)
- **+100 tokens per turn**

### Example 3-step episode:
- Before: 300 (system) + 100 + 50 + 50 = **500 tokens**
- After: 300 (system) + 200 + 150 + 150 = **800 tokens**
- **+300 tokens per episode** (~60% increase)

## Trade-offs

### ✅ Advantages:
- More explicit tool availability
- Better for models that don't follow system prompts well
- Tools are in immediate attention window
- Reduces ambiguity about what actions are available
- Easier for model to copy exact syntax

### ⚠️ Considerations:
- Uses more tokens (~60% increase)
- Some redundancy with system prompt
- Longer context per turn

## When to Use This Approach

**Use this (tools in user messages) when:**
- Model struggles with tool usage
- You want maximum clarity
- Token budget allows
- Model has attention issues with system prompts

**Use system-only approach when:**
- Token efficiency is critical
- Model follows system prompts well
- Context length is limited
- You want minimal repetition

## Expected Debug Output

With the new structure, you'll see:

```
🔷 SYSTEM MESSAGE (Step 0 - Shown Once)
You are an AI assistant...
[Full tool descriptions]

👤 USER MESSAGE (Step 0 - Turn 1)
TASK: Create Jira ticket...
--- Turn 1 ---
[IMAGE]
Available tools:
• browser(action="left_click"...)
• browser(action="right_click"...)
[...all 10 tools...]
Actions remaining: 10
Your action:

🤖 ASSISTANT RESPONSE (Step 0)
browser(action="left_click", x=250, y=150)

👤 USER MESSAGE (Step 1 - Turn 2)
--- Turn 2 ---
[IMAGE]
Available tools:  ← TOOLS REPEATED HERE
• browser(action="left_click"...)
• browser(action="right_click"...)
[...all 10 tools...]
Actions remaining: 9
Your action:
```

## Recommendation

For your browser automation task, **this is a good choice** because:
1. Browser actions require precise syntax
2. Having tools visible helps model use correct format
3. 10 tools is manageable (~100 tokens)
4. The extra tokens are worth the clarity

If you find it works well without repetition, you can always switch back to system-only by reverting the `next_step_template` change.

