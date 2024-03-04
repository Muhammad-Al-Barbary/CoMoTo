from multimodal_breast_analysis.engine.engine import Engine
from multimodal_breast_analysis.configs.configs import load_configs
from multimodal_breast_analysis.engine.evaluate import dbt_final_eval, mammo_final_eval
import argparse


def main(args):
    config = load_configs(args.config_name)
    engine = Engine(config)
    if args.mammo:
        engine.warmup()
        engine.load(mode="teacher", path=config.networks["best_teacher_cp"])
        print("\n\TESTING METRICS:")
        print(engine.test('teacher', 'testing'))
        print(mammo_final_eval(engine))
    if args.dbt:
        engine.load(mode="teacher", path=config.networks["best_teacher_cp"])
        engine.load(mode="student", path=config.networks["best_teacher_cp"])
        engine.train()
        print("\n\TESTNIG METRICS:")
        engine.load(mode="student", path=config.networks["best_student_cp"])
        print(dbt_final_eval(engine, pred_csv = "best_"+args.config_name+'.csv', temp_path="temp_"+args.config_name+'/'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Multimodal Breast Lesion Detection')
    parser.add_argument(
                '--config_name', type = str, default = None,
                help = 'name of the config file to be loaded (without extension)'
                )
    parser.add_argument(
                '--mammo', action='store_true',
                help = 'train mammography model'
                )
    parser.add_argument(
                '--dbt', action='store_true',
                help = 'train dbt model'
                )
    args = parser.parse_args()
    main(args)