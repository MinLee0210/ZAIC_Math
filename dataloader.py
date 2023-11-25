from datasets import Dataset
from dataclasses import dataclass
from typing import Optional, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy

option_to_index = {option: idx for idx, option in enumerate('ABCD')}
index_to_option = {v: k for k,v in option_to_index.items()}
dict_one_hot_answer = {
    0: "A",
    1: "B",
    2: "C",
    3: "D"
}

# ----- Dataset >> Pandas -----
def to_train_df(json_data): 
    list_question = []
    list_answer = []
    list_A = []
    list_B = []
    list_C = []
    list_D = []
    list_explanation = []

    for record in json_data['data']:
        question = record['question']
        choices = record['choices']
        try:
            explanation = record['explanation']
        except KeyError:
            explanation = "None"
        answer = record['answer']

        list_A.append(choices[0])
        list_B.append(choices[1])
        list_C.append(choices[2])

        try:
            list_D.append(choices[3])
        except IndexError:
            list_D.append("None")
        list_question.append(question)
        one_hot_answer = choices.index(answer)
        list_answer.append(dict_one_hot_answer[one_hot_answer])
        list_explanation.append(explanation)

    data_df = pd.DataFrame(list(zip(list_question, list_explanation, list_A, list_B, list_C, list_D, list_answer)),
                       columns=['question', 'explanation', 'A', 'B', 'C', 'D', 'answer'])

    return data_df

def to_test_df(json_data): 
    dict_one_hot_answer = {
    0: "A",
    1: "B",
    2: "C",
    3: "D"
}

    list_id = []
    list_question = []
    list_A = []
    list_B = []
    list_C = []
    list_D = []

    for record in json_data['data']:
    id = record['id']
    question = record['question']
    choices = record['choices']
    try:
        explanation = record['explanation']
    except KeyError:
        explanation = "None"

    list_A.append(choices[0])
    list_B.append(choices[1])
    list_C.append(choices[2])
    try:
        list_D.append(choices[3])
    except IndexError:
        list_D.append("None")
    list_question.append(question)
    list_id.append(id)
    data_df = pd.DataFrame(list(zip(list_id, list_question, list_A, list_B, list_C, list_D)),
                       columns=['id', 'question', 'A', 'B', 'C', 'D',])
    return data_df

# ----- Pre-processing -----
def preprocess(example):
    first_sentence = [ "[CLS] " + example['explanation'] ] * 4
    second_sentences = [" #### " + example['question'] + " [SEP] " + example[option] + " [SEP]" for option in 'ABCD']
    tokenized_example = tokenizer(first_sentence, second_sentences, truncation=True,
                                  max_length=MAX_INPUT, add_special_tokens=False)
    tokenized_example['label'] = option_to_index[example['answer']]

    return tokenized_example

def test_preprocess(example):
    first_sentence = [ "[CLS] " + example['explanation'] ] * 4
    second_sentences = [" #### " + example['question'] + " [SEP] " + example[option] + " [SEP]" for option in 'ABCD']
    tokenized_example = tokenizer(first_sentence, second_sentences, truncation=True,
                                  max_length=MAX_INPUT, add_special_tokens=False, padding='max_length')

    return tokenized_example

def create_dataset(df_train, df_valid): 
    dataset_valid = Dataset.from_pandas(df_valid)
    dataset = Dataset.from_pandas(df_train)
    dataset = dataset.remove_columns(["__index_level_0__"])
    return dataset, dataset_valid

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch
