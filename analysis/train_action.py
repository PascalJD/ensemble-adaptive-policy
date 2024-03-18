if __name__ == '__main__':
    from ml_logger import logger, instr
    from analysis import RUN
    import jaynes
    from scripts.train_action import main
    from config.action import Config
    from params_proto.neo_hyper import Sweep

    sweep = Sweep(RUN, Config).load("/home/pjutrasd/depot_symlink/projects/ensemble-adaptive-policy/analysis/json/action.jsonl")

    for kwargs in sweep:
        logger.print(RUN.prefix, color='green')
        jaynes.config("local")
        thunk = instr(main, **kwargs)
        jaynes.run(thunk)

    jaynes.listen()
