# Contributing

Contributions are welcome.

## What's useful
- Adding more Indian car models to `core/panel_solver.py`
- Improving the WLTP reconstruction against the official 1Hz trace
- Adding test coverage for `modifications.py` and `optimizer.py`
- Reporting validation errors against wind tunnel or manufacturer data

## How to contribute
1. Fork the repository
2. Create a branch: `git checkout -b your-feature-name`
3. Make your changes
4. Push and open a pull request

## Ground rules
- Every physics constant must have a source citation in the code comment
- No machine learning, no black-box estimations
- If you add a car, include the manufacturer Cd reference and source