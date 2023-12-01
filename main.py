from multimodal_breast_analysis.engine.engine import Engine
import wandb

def main():
    engine = Engine()
    engine.warmup()
    engine.train()
    engine.load("student")
    engine.test("student")
    wandb.finish()


if __name__ == "__main__":
    main()