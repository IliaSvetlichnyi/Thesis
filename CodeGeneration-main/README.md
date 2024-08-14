# CodeGeneration

## Dependencies

```bash
pip install -r requirements.txt
```

## Before run

Put the credentials into: `src/env.json`

```javascript
{"OPENROUTER_API_KEY": "YOUR_API_KEY"}
```

## Running jupyter notebook main

```bash
cd src
python3 main.ipynb
```

## Runnig cmd util

```bash
cd src
python3 main_controller_cmd.py --dataset ../datasets/insurance.csv --parameters_file ../cases/simple/parameters.json --steps_file ../cases/simple/steps.json -o out
```

