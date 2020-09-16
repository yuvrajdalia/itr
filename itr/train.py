from time import time
from pathlib import Path
import torch
from transformers import BertConfig, BertModel, BertForMaskedLM, BertTokenizer
from pytorch_lightning import Trainer
from easydict import EasyDict as ED
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import IndicDataset, PadSequence
from config import replace, preEnc, preEncDec

#this function will preprocess data fron .txt file and create train and val .csv files
def preproc_data():
    from data import split_data
    split_data('../hin-eng/hin.txt', '../hin-eng')

#create the main class which involves all the components for pytorch lightning to run
class TranslationModel(pl.LightningModule):

    def __init__(self,config):

        super().__init__() 
        src_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        tgt_tokenizer.bos_token = '<s>'
        tgt_tokenizer.eos_token = '</s>'
        #hidden_size and intermediate_size are both wrt all the attention heads. 
        #Should be divisible by num_attention_heads
        encoder_config = BertConfig(vocab_size=src_tokenizer.vocab_size,
                                    hidden_size=config.hidden_size,
                                    num_hidden_layers=config.num_hidden_layers,
                                    num_attention_heads=config.num_attention_heads,
                                    intermediate_size=config.intermediate_size,
                                    hidden_act=config.hidden_act,
                                    hidden_dropout_prob=config.dropout_prob,
                                    attention_probs_dropout_prob=config.dropout_prob,
                                    max_position_embeddings=512,
                                    type_vocab_size=2,
                                    initializer_range=0.02,
                                    layer_norm_eps=1e-12)

        decoder_config = BertConfig(vocab_size=tgt_tokenizer.vocab_size,
                                    hidden_size=config.hidden_size,
                                    num_hidden_layers=config.num_hidden_layers,
                                    num_attention_heads=config.num_attention_heads,
                                    intermediate_size=config.intermediate_size,
                                    hidden_act=config.hidden_act,
                                    hidden_dropout_prob=config.dropout_prob,
                                    attention_probs_dropout_prob=config.dropout_prob,
                                    max_position_embeddings=512,
                                    type_vocab_size=2,
                                    initializer_range=0.02,
                                    layer_norm_eps=1e-12,)

        #Create encoder and decoder embedding layers.
        encoder_embeddings = torch.nn.Embedding(src_tokenizer.vocab_size, config.hidden_size, padding_idx=src_tokenizer.pad_token_id)
        decoder_embeddings = torch.nn.Embedding(tgt_tokenizer.vocab_size, config.hidden_size, padding_idx=tgt_tokenizer.pad_token_id)

        encoder = BertModel(encoder_config)
        encoder.set_input_embeddings(encoder_embeddings.cuda())
        
        #decoder_config.add_cross_attention=True
        #decoder_config.is_decoder=True
        decoder = BertForMaskedLM(decoder_config)
        decoder.set_input_embeddings(decoder_embeddings.cuda())
        #Creating encoder and decoder with their respective embeddings.
        tokenizers = ED({'src': src_tokenizer, 'tgt': tgt_tokenizer})
        self.encoder = encoder
        self.decoder = decoder
        self.pad_sequence=PadSequence(tokenizers.src.pad_token_id, tokenizers.tgt.pad_token_id)
        self.tokenizers=tokenizers
        self.config=config

    def forward(self, encoder_input_ids, decoder_input_ids):

        encoder_hidden_states = self.encoder(encoder_input_ids)[0]
        loss, logits = self.decoder(decoder_input_ids,
                                    encoder_hidden_states=encoder_hidden_states, 
                                    labels=decoder_input_ids)

        return loss, logits

    
    def configure_optimizers(self):
      return torch.optim.Adam(self.parameters(), lr=self.config.lr)
    
    def training_step(self,batch,batch_idx):
      source=batch[0]
      target=batch[1]
      loss,logits=self(source,target)
      tensorboard_logs={'train_loss':loss}
      return {'loss':loss,'log':tensorboard_logs}
    
    def train_dataloader(self):
      train_loader = DataLoader(IndicDataset(self.tokenizers.src, self.tokenizers.tgt, self.config.data, True), 
                            batch_size=self.config.batch_size, 
                            shuffle=False, 
                            collate_fn=self.pad_sequence)
      return train_loader

    def validation_step(self,batch,batch_idx):
      source=batch[0]
      target=batch[1]
      loss,logits=self(source,target)
      return {'val_loss':loss}

    def val_dataloader(self):
      val_loader = DataLoader(IndicDataset(self.tokenizers.src, self.tokenizers.tgt, self.config.data, False), 
                            batch_size=self.config.batch_size, 
                            shuffle=False, 
                            collate_fn=self.pad_sequence)
      return val_loader
    
    def validation_epoch_end(self,outputs):
      avg_loss=torch.stack([x['val_loss'] for x in outputs]).mean()
      tensorboard_logs={'avg_val_loss':avg_loss}
      return {'val_loss':avg_loss,'log':tensorboard_logs}



def main():
    rconf = preEncDec
    trainer=Trainer(gpus=1,max_epochs=3)
    model=TranslationModel(rconf)
    trainer.fit(model)


if __name__ == '__main__':
    #preproc_data()
    main()








