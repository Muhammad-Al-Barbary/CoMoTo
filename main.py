from multimodal_breast_analysis.engine.engine import Engine
import wandb

def main():
    engine = Engine()
    # engine.load('teacher', 'checkpoints/best_teacher_res50_b10_aug0.75_cocoall.pt')
    # engine.load('student', 'checkpoints/best_teacher_res50_b10_aug0.75_cocoall.pt')
    # print(engine.test("teacher", "testing"))
    engine.warmup()
    engine.train()
    engine.load("student")
    engine.test("student", "testing")
    wandb.finish()


if __name__ == "__main__":
    main()