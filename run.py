import argparse

from src.multilingual_stories_topic_space.main import main

parser = argparse.ArgumentParser(description="just something to see if I want only to render the graph or not")
if __name__ == "__main__":
    parser.add_argument("--graph", type=bool, default=True, help="Only render graph")
    args = parser.parse_args()
    main(args.graph)
    print("finished")
