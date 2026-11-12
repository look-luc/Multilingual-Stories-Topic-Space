import argparse

from src.multilingual_stories_topic_space.main import main

parser = argparse.ArgumentParser(description="just something to see if I want only to render the graph or not")
if __name__ == "__main__":
    parser.add_argument("--graph", action='store_true', default=False, help="Only render graph")
    parser.add_argument("--lda", action='store_true', default=False, help="LDA traier")
    parser.add_argument("--parse_out", action='store_true', default=False, help="Parsing output from LDA run")
    args = parser.parse_args()
    main(args.graph, args.lda)
    print("finished")
