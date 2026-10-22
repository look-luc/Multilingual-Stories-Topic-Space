import src.model.model as get_model
from src.data.get_data import get_all_data, get_some_data


def setup(data_amount:str="some"):
    if data_amount.lower() == "some":
        assamese_data, vietnamese_data, acholi_data = get_some_data()
        return assamese_data, vietnamese_data, acholi_data
    elif data_amount.lower() == "all":
        raise Exception("TBD")
    else:
        raise Exception("Either: 'some' or 'all'")

def main():
    lda_model = get_model.LDA()
    roberta = get_model.XLM_RoBerta()

    data_1, data_2, data_3 = setup()

if __name__=="__main__":
    print("ran")
