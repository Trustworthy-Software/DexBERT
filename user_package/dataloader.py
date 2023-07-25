
from utils import truncate_tokens

class PreprocessEmbedding():
    def __init__(self, indexer, max_len=512):
        super().__init__()
        self.max_len = max_len
        self.indexer = indexer # function from token to token index
    
    def __call__(self, instance: tuple):
        tokens, class_id = instance

        truncate_tokens(tokens, self.max_len)

        segment_ids = [0]*len(tokens)
        input_mask  = [1]*len(tokens)

        # Token Indexing
        input_ids = self.indexer(tokens)

         # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        input_mask.extend([0]*n_pad)
    
        return (input_ids, segment_ids, input_mask, class_id)
