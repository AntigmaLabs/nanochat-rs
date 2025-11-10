# Fix Until Fixture Test Passes

This command runs the fixture test in a loop until it passes, automatically fixing any compilation errors or test failures.

this is focused on gpt.rs interence implementation of the python version reference/nanogpt/ where reference/ is a uv environment

## What it does
1. Runs `cargo test --test fixture_test` to check if the test passes
2. If the test fails, analyzes the error output
3. Attempts to fix the issues found
4. Repeats until the test passes or maximum iterations reached

## Implementation
The agent will:
- Run the fixture test and capture output
- Parse error messages to identify specific issues
- Deep dive and think hard on what could the source of bug
- If needed devise new test to narrow down the issute

## Safety
- Maximum of 20 iterations to prevent infinite loops
- Provides clear progress updates
- Stops on critical errors that can't be auto-fixed
