import torch
from transformers import BertTokenizer, BertConfig
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np

class bert_load():
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        self.review = None
        self.dataloader_val=None
    def get_predict(self, review):
        self.review = review
        self.load_model()
        self.preprocess_text()
        label = self.evaluate()
        return label[0]
    def load_model(self):
        # model = BertForSequenceClassification.from_pretrained(
        #     'bert-base-uncased',
        #     num_labels = 2,
        #     output_attentions = True,
        #     output_hidden_states = False
        #     )
        config = BertConfig.from_json_file('./model_save/config.json')
        model = BertForSequenceClassification.from_pretrained(self.output_dir, config=config)
        tokenizer = BertTokenizer.from_pretrained(self.output_dir)
        self.model=model
        self.tokenizer=tokenizer

    def preprocess_text(self):
        encoded_review = self.tokenizer.batch_encode_plus(
            self.review,
            add_special_tokens = True,
            return_attention_mask = True,
            padding =True,
            max_length = 256,
            return_tensors = 'pt'
        )
        input_ids_val = encoded_review['input_ids']
        attention_masks_val = encoded_review['attention_mask']
        dataset_val = TensorDataset(input_ids_val,attention_masks_val) #type: TensorDataset
        dataloader_val = DataLoader(dataset_val, batch_size=1) #type: DataLoader
        self.dataloader_val=dataloader_val

    def evaluate(self):
        self.model.eval()
        predictions, true_vals = [], []
        for batch in self.dataloader_val:
            inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1]
                        }
            with torch.no_grad():        
                outputs = self.model(**inputs)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            predictions.append(logits)
        label = np.argmax(predictions[0], axis =1 )

        return label
# #save the model
# import os

# # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
# output_dir = './model_save/'

# # Create output directory if needed
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# print("Saving model to %s" % output_dir)

# # Save a trained model, configuration and tokenizer using `save_pretrained()`.
# # They can then be reloaded using `from_pretrained()`
# model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
# model_to_save.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir)

# Good practice: save your training arguments together with the trained model
# torch.save(args, os.path.join(output_dir, 'training_args.bin'))
if __name__=="__main__":
    output_dir = "./model_save/"
    # bert_test = bert_load(output_dir)
    # review = "Basically there's a family where a little boy ...	"
    # label=bert_test.get_predict(review)
    # print(label)
    





