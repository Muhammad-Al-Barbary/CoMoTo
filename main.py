from multimodal_breast_analysis.engine.engine import Engine
def main():
    engine = Engine()
    engine.warmup()
    engine.train()
    engine.load("student")
    engine.test("student")



if __name__ == "__main__":
    main()