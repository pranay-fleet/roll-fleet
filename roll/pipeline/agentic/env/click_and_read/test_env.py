#!/usr/bin/env python3
"""
Simple test script for the Click and Read environment
Run this to verify the environment works correctly before training
"""

import logging
from env import ClickAndReadEnv
from PIL import Image

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_successful_episode():
    """Test a successful episode where agent clicks and submits correctly"""
    logger.info("=" * 60)
    logger.info("TEST 1: Successful Episode")
    logger.info("=" * 60)
    
    env = ClickAndReadEnv()
    obs, info = env.reset(seed=42)
    
    logger.info(f"Initial observation shape: {obs.shape}")
    logger.info(f"Env instruction: {info['env_instruction'][:100]}...")
    
    # Save initial image
    Image.fromarray(obs).save("test_output_step0_button.png")
    logger.info("Saved initial button image to: test_output_step0_button.png")
    
    # Click the button (we know it's at top_left for this seed)
    obs, reward, terminated, truncated, info = env.step("<answer>click:[12, 12]</answer>")
    logger.info(f"Step 1 - Click result: reward={reward}, terminated={terminated}, info={info['metrics']}")
    
    # Save image after click
    Image.fromarray(obs).save("test_output_step1_word.png")
    logger.info("Saved word reveal image to: test_output_step1_word.png")
    
    # Submit the correct answer
    obs, reward, terminated, truncated, info = env.step("<answer>submit:APPLE</answer>")
    logger.info(f"Step 2 - Submit result: reward={reward}, terminated={terminated}, info={info['metrics']}")
    
    assert reward == 1.0, f"Expected reward 1.0, got {reward}"
    assert info['metrics']['success'], "Expected success to be True"
    logger.info("✓ Test passed: Successful episode")
    print()


def test_failed_click():
    """Test episode where agent clicks wrong location"""
    logger.info("=" * 60)
    logger.info("TEST 2: Failed Click")
    logger.info("=" * 60)
    
    env = ClickAndReadEnv()
    obs, info = env.reset(seed=42)
    
    # Click wrong location
    obs, reward, terminated, truncated, info = env.step("<answer>click:[50, 50]</answer>")
    logger.info(f"Step 1 - Wrong click result: reward={reward}, button_clicked={info['metrics']['button_clicked']}")
    
    # Try to submit anyway
    obs, reward, terminated, truncated, info = env.step("<answer>submit:APPLE</answer>")
    logger.info(f"Step 2 - Submit result: reward={reward}, terminated={terminated}")
    
    assert reward == 0.0, f"Expected reward 0.0, got {reward}"
    assert not info['metrics']['button_clicked'], "Expected button_clicked to be False"
    logger.info("✓ Test passed: Failed click correctly handled")
    print()


def test_wrong_answer():
    """Test episode where agent submits wrong word"""
    logger.info("=" * 60)
    logger.info("TEST 3: Wrong Answer")
    logger.info("=" * 60)
    
    env = ClickAndReadEnv()
    obs, info = env.reset(seed=42)
    
    # Click correctly
    obs, reward, terminated, truncated, info = env.step("<answer>click:[12, 12]</answer>")
    logger.info(f"Step 1 - Click result: button_clicked={info['metrics']['button_clicked']}")
    
    # Submit wrong answer
    obs, reward, terminated, truncated, info = env.step("<answer>submit:BANANA</answer>")
    logger.info(f"Step 2 - Wrong submit result: reward={reward}, terminated={terminated}")
    
    assert reward == 0.0, f"Expected reward 0.0, got {reward}"
    assert not info['metrics']['success'], "Expected success to be False"
    logger.info("✓ Test passed: Wrong answer correctly handled")
    print()


def test_invalid_format():
    """Test episode with invalid action format"""
    logger.info("=" * 60)
    logger.info("TEST 4: Invalid Format")
    logger.info("=" * 60)
    
    env = ClickAndReadEnv()
    obs, info = env.reset(seed=42)
    
    # Invalid format
    obs, reward, terminated, truncated, info = env.step("click at 12, 12")
    logger.info(f"Step 1 - Invalid format result: reward={reward}, action_valid={info['metrics']['action_is_valid']}")
    
    assert reward == -0.01, f"Expected format penalty -0.01, got {reward}"
    assert not info['metrics']['action_is_valid'], "Expected action_is_valid to be False"
    logger.info("✓ Test passed: Invalid format correctly penalized")
    print()


def test_max_steps():
    """Test episode that reaches max steps"""
    logger.info("=" * 60)
    logger.info("TEST 5: Max Steps Timeout")
    logger.info("=" * 60)
    
    env = ClickAndReadEnv(max_steps=2)
    obs, info = env.reset(seed=42)
    
    # Click correctly
    obs, reward, terminated, truncated, info = env.step("<answer>click:[12, 12]</answer>")
    logger.info(f"Step 1 - Click result: terminated={terminated}")
    
    # Wait until max steps (don't submit)
    obs, reward, terminated, truncated, info = env.step("<answer>click:[50, 50]</answer>")
    logger.info(f"Step 2 - Max steps reached: reward={reward}, terminated={terminated}")
    
    assert terminated, "Expected terminated to be True"
    assert reward == 0.0, f"Expected reward 0.0 for timeout, got {reward}"
    logger.info("✓ Test passed: Max steps timeout correctly handled")
    print()


if __name__ == "__main__":
    logger.info("\n" + "=" * 60)
    logger.info("Running Click and Read Environment Tests")
    logger.info("=" * 60 + "\n")
    
    test_successful_episode()
    test_failed_click()
    test_wrong_answer()
    test_invalid_format()
    test_max_steps()
    
    logger.info("=" * 60)
    logger.info("ALL TESTS PASSED ✓")
    logger.info("=" * 60)
    logger.info("\nCheck the generated images:")
    logger.info("  - test_output_step0_button.png (initial state with green button)")
    logger.info("  - test_output_step1_word.png (after successful click, showing word)")

