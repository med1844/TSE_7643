from main import TrainArgs

with open("configs/default.json", "w") as f:
    default_args = TrainArgs.default()
    f.write(default_args.model_dump_json(indent=4))
