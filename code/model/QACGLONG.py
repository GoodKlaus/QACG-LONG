from __future__ import absolute_import, division, print_function

import copy
import json
import math
import collections
import torch.utils.checkpoint

import inspect
import six
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import logging
from functools import partial

from util.lrp import *

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# access global vars here
global func_inputs
global func_activations
func_inputs = collections.defaultdict(list)
func_activations = collections.defaultdict(list)

def get_inputivation(name):
    def hook(model, input, output):
        func_inputs[name] = [_in for _in in input]
    return hook

def get_activation(name):
    def hook(model, input, output):
        func_activations[name] = output
    return hook

def get_activation_multi(name):
    def hook(model, input, output):
        func_activations[name] = [_out for _out in output]
    return hook

# TODO: make this init as a part of the model init
def init_hooks_lrp(model):
    """
    Initialize all the hooks required for full lrp for BERT model.
    """
    # in order to backout all the lrp through layers
    # you need to register hooks here.
    model.classifier.register_forward_hook(
        get_inputivation('model.classifier'))
    model.classifier.register_forward_hook(
        get_activation('model.classifier'))

    model.bert.embeddings.register_forward_hook(
        get_activation('model.bert.embeddings'))

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BERTIntermediate1(nn.Module):
    def __init__(self, config):
        super(BERTIntermediate1, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class ContextBERTSelfAttention1(nn.Module):
    def __init__(self, config, layer_id):
        super(ContextBERTSelfAttention1, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob
        
        # separate projection layers for tokens with global attention
        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)

        # learnable context integration factors
        # enforce initialization to zero as to leave the pretrain model
        # unperturbed in the beginning
        self.context_for_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.context_for_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.lambda_q_context_layer = nn.Linear(self.head_dim, 1)
        self.lambda_q_query_layer = nn.Linear(self.head_dim, 1)
        self.lambda_k_context_layer = nn.Linear(self.head_dim, 1)
        self.lambda_k_key_layer = nn.Linear(self.head_dim, 1)

        # zero-centered activation function, specifically for re-arch fine tunning
        self.lambda_act = nn.Sigmoid()
        self.quasi_act = nn.Sigmoid()
        
        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self.one_sided_attn_window_size = attention_window // 2


    def forward(self, hidden_states, attention_mask,
                # optional parameters for saving context information
                device=None, 
                layer_head_mask=None,
                is_index_masked=None,
                is_index_global_attn=None,
                is_global_attn=None,
                context_embedded=None):
        hidden_states = hidden_states.transpose(0, 1)
        context_embedded = context_embedded.transpose(0, 1)

        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)

        # Project context information
        context_embedded_q = self.context_for_q(context_embedded)
        context_embedded_k = self.context_for_k(context_embedded)

        seq_len, batch_size, embed_dim = hidden_states.size()
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)
        query_vectors = query_vectors.view(
            seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        key_vectors = key_vectors.view(
            seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        
        context_embedded_q /= math.sqrt(self.head_dim)

        context_embedded_q = context_embedded_q.view(
            seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        context_embedded_k = context_embedded_k.view(
            seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        attn_scores = self._sliding_chunks_query_key_matmul(
            query_vectors, key_vectors, self.one_sided_attn_window_size
        )
        
        # Quasi-attention scores
        quasi_attn_scores = self._sliding_chunks_query_key_matmul(
            context_embedded_q, context_embedded_q, self.one_sided_attn_window_size
        )

        # values to pad for attention probs
        remove_from_windowed_attention_mask = (
            attention_mask != 0)[:, :, None, None]

        # cast to fp32/fp16 then replace 1's with -inf
        float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
            remove_from_windowed_attention_mask, -10000.0
        )
        # diagonal mask with zeros everywhere and -inf inplace of padding
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_mask.new_ones(size=float_mask.size()
                                ), float_mask, self.one_sided_attn_window_size
        )

        # pad local attention probs
        attn_scores += diagonal_mask
        quasi_attn_scores += diagonal_mask
        
        assert list(attn_scores.size()) == [
            batch_size,
            seq_len,
            self.num_heads,
            self.one_sided_attn_window_size * 2 + 1,
        ], f"local_attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads}, {self.one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.size()}"
        
        # compute local attention probs from global attention keys and contact over window dim
        if is_global_attn:
            # compute global attn indices required through out forward fn
            (
                max_num_global_attn_indices,
                is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero,
            ) = self._get_global_attn_indices(is_index_global_attn)
            # calculate global attn probs from global key

            global_key_attn_scores = self._concat_with_global_key_attn_probs(
                query_vectors=query_vectors,
                key_vectors=key_vectors,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            )

            # Compute Quasi-attention scores - context query and context key with only global attention indices.
            global_key_quasi_attn_scores = self._concat_with_global_key_attn_probs(
                query_vectors=context_embedded_q,
                key_vectors=context_embedded_k,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            )
            # concat to local_attn_probs
            # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
            attn_scores = torch.cat(
                (global_key_attn_scores, attn_scores), dim=-1)
            quasi_attn_scores = torch.cat(
                (global_key_quasi_attn_scores, quasi_attn_scores), dim=-1)

            # free memory
            del global_key_attn_scores
            del global_key_quasi_attn_scores
        
        attn_probs = nn.functional.softmax(
            attn_scores, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability

        # Obtain Quasi-attention probabilities
        quasi_scalar = 1.0
        quasi_attention_scores = 1.0 * quasi_scalar * \
            self.quasi_act(quasi_attn_scores)
        lambda_q_context = self.lambda_q_context_layer(context_embedded_q)
        lambda_q_query = self.lambda_q_query_layer(
            query_vectors*math.sqrt(self.head_dim))
        lambda_q = self.quasi_act(lambda_q_context + lambda_q_query)
        lambda_k_context = self.lambda_k_context_layer(context_embedded_k)
        lambda_k_key = self.lambda_k_key_layer(key_vectors)
        lambda_k = self.quasi_act(lambda_k_context + lambda_k_key)
        lambda_q_scalar = 1.0
        lambda_k_scalar = 1.0
        lambda_context = lambda_q_scalar*lambda_q + lambda_k_scalar*lambda_k
        lambda_context = (1 - lambda_context)
        quasi_attn_probs = lambda_context * quasi_attention_scores
        
        new_attn_probs = attn_probs + quasi_attn_probs

        if layer_head_mask is not None:
            assert layer_head_mask.size() == (
                self.num_heads,
            ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
            new_attn_probs = layer_head_mask.view(1, 1, -1, 1) * new_attn_probs

        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        new_attn_probs = torch.masked_fill(
            new_attn_probs, is_index_masked[:, :, None, None], 0.0)
        new_attn_probs = new_attn_probs.type_as(attn_scores)

        # free memory
        del attn_scores
        del attn_probs
        del quasi_attn_scores
        del quasi_attn_probs
        
        # apply dropout
        new_attn_probs = nn.functional.dropout(
            new_attn_probs, p=self.dropout, training=self.training)

        value_vectors = value_vectors.view(
            seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        # compute local attention output with global attention value and add
        if is_global_attn:
            # compute sum of global and local attn
            attn_output = self._compute_attn_output_with_global_indices(
                value_vectors=value_vectors,
                attn_probs=new_attn_probs,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            )
        else:
            # compute local attn only
            attn_output = self._sliding_chunks_matmul_attn_probs_value(
                new_attn_probs, value_vectors, self.one_sided_attn_window_size
            )

        assert attn_output.size() == (batch_size, seq_len, self.num_heads,
                                      self.head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(
            seq_len, batch_size, embed_dim).contiguous()
        
        if is_global_attn:
            global_attn_output, global_attn_probs = self._compute_global_attn_output_from_hidden(
                hidden_states=hidden_states,
                context_embedded=context_embedded,
                max_num_global_attn_indices=max_num_global_attn_indices,
                layer_head_mask=layer_head_mask,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                is_index_masked=is_index_masked,
            )

            # get only non zero global attn output
            nonzero_global_attn_output = global_attn_output[
                is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]
            ]

            # overwrite values with global attention
            attn_output[is_index_global_attn_nonzero[::-1]] = nonzero_global_attn_output.view(
                len(is_local_index_global_attn_nonzero[0]), -1
            )
            # The attention weights for tokens with global attention are
            # just filler values, they were never used to compute the output.
            # Fill with 0 now, the correct values are in 'global_attn_probs'.
            new_attn_probs[is_index_global_attn_nonzero] = 0
            
            del global_attn_output
            del global_attn_probs
            
        attn_output = attn_output.transpose(0, 1)

        return attn_output, new_attn_probs
    
    @staticmethod
    def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
        """pads rows and then flips rows and columns"""
        hidden_states_padded = nn.functional.pad(
            hidden_states_padded, padding
        )  # padding value is not important because it will be overwritten
        hidden_states_padded = hidden_states_padded.view(
            *hidden_states_padded.size()[:-2], hidden_states_padded.size(-1), hidden_states_padded.size(-2)
        )
        return hidden_states_padded

    @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states):
        """
        shift every row 1 step right, converting columns into diagonals.
        Example:
        ```python
        chunked_hidden_states: [
            0.4983,
            2.6918,
            -0.0071,
            1.0492,
            -1.8348,
            0.7672,
            0.2986,
            0.0285,
            -0.7584,
            0.4206,
            -0.0405,
            0.1599,
            2.0514,
            -1.1600,
            0.5372,
            0.2629,
        ]
        window_overlap = num_rows = 4
        ```
                     (pad & diagonalize) => [ 0.4983, 2.6918, -0.0071, 1.0492, 0.0000, 0.0000, 0.0000
                       0.0000, -1.8348, 0.7672, 0.2986, 0.0285, 0.0000, 0.0000 0.0000, 0.0000, -0.7584, 0.4206,
                       -0.0405, 0.1599, 0.0000 0.0000, 0.0000, 0.0000, 2.0514, -1.1600, 0.5372, 0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = chunked_hidden_states.size()
        chunked_hidden_states = nn.functional.pad(
            chunked_hidden_states, (0, window_overlap + 1)
        )  # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1). Padding value is not important because it'll be overwritten
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, -1
        )  # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        chunked_hidden_states = chunked_hidden_states[
            :, :, :-window_overlap
        ]  # total_num_heads x num_chunks x window_overlap*window_overlap
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        return chunked_hidden_states

    @staticmethod
    def _chunk(hidden_states, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""

        # non-overlapping chunks of size = 2w
        hidden_states = hidden_states.view(
            hidden_states.size(0),
            hidden_states.size(1) // (window_overlap * 2),
            window_overlap * 2,
            hidden_states.size(2),
        )

        # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
        chunk_size = list(hidden_states.size())
        chunk_size[1] = chunk_size[1] * 2 - 1

        chunk_stride = list(hidden_states.stride())
        chunk_stride[1] = chunk_stride[1] // 2
        return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len) -> torch.Tensor:
        beginning_mask_2d = input_tensor.new_ones(
            affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:,
                                       :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        # `== 1` converts to bool or uint8
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))
        ending_input = input_tensor[:, -
                                    affected_seq_len:, :, -(affected_seq_len + 1):]
        ending_mask = ending_mask.expand(ending_input.size())
        # `== 1` converts to bool or uint8
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))

    def _sliding_chunks_query_key_matmul(self, query: torch.Tensor, key: torch.Tensor, window_overlap: int):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This
        implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer) with an
        overlap of size window_overlap
        """
        batch_size, seq_len, num_heads, head_dim = query.size()
        assert (
            seq_len % (window_overlap * 2) == 0
        ), f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1

        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size window_overlap * 2
        query = query.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len, head_dim)
        key = key.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len, head_dim)

        query = self._chunk(query, window_overlap)
        key = self._chunk(key, window_overlap)

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        diagonal_chunked_attention_scores = torch.einsum(
            "bcxd,bcyd->bcxy", (query, key))  # multiply

        # convert diagonals into columns
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.

        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1,
             window_overlap, window_overlap * 2 + 1)
        )

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, -1, window_overlap:, : window_overlap + 1
        ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
            :, :, -(window_overlap + 1): -1, window_overlap + 1:
        ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
            :, 0, : window_overlap - 1, 1 - window_overlap:
        ]

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(
        self, attn_probs: torch.Tensor, value: torch.Tensor, window_overlap: int
    ):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        batch_size, seq_len, num_heads, head_dim = value.size()

        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size()[:3] == value.size()[:3]
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1
        )

        # group batch_size and num_heads dimensions into one
        value = value.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len, head_dim)

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = nn.functional.pad(
            value, (0, 0, window_overlap, window_overlap), value=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (batch_size * num_heads,
                              chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(
            size=chunked_value_size, stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum(
            "bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)

    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        """compute global attn indices required throughout forward pass"""
        # helper variable
        num_global_attn_indices = is_index_global_attn.long().sum(dim=1)

        # max number of global attn indices in batch
        max_num_global_attn_indices = num_global_attn_indices.max()

        # indices of global attn
        is_index_global_attn_nonzero = is_index_global_attn.nonzero(
            as_tuple=True)

        # helper variable
        is_local_index_global_attn = torch.arange(
            max_num_global_attn_indices, device=is_index_global_attn.device
        ) < num_global_attn_indices.unsqueeze(dim=-1)

        # location of the non-padding values within global attention indices
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(
            as_tuple=True)

        # location of the padding values within global attention indices
        is_local_index_no_global_attn_nonzero = (
            is_local_index_global_attn == 0).nonzero(as_tuple=True)
        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )

    def _concat_with_global_key_attn_probs(
        self,
        key_vectors,
        query_vectors,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
    ):
        batch_size = key_vectors.shape[0]

        # create only global key vectors
        key_vectors_only_global = key_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim
        )

        key_vectors_only_global[is_local_index_global_attn_nonzero] = key_vectors[is_index_global_attn_nonzero]

        # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = torch.einsum(
            "blhd,bshd->blhs", (query_vectors, key_vectors_only_global))

        attn_probs_from_global_key[
            is_local_index_no_global_attn_nonzero[0], :, :, is_local_index_no_global_attn_nonzero[1]
        ] = -10000.0

        return attn_probs_from_global_key

    def _compute_attn_output_with_global_indices(
        self,
        value_vectors,
        attn_probs,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
    ):
        batch_size = attn_probs.shape[0]

        # cut local attn probs to global only
        attn_probs_only_global = attn_probs.narrow(
            -1, 0, max_num_global_attn_indices)
        # get value vectors for global only
        value_vectors_only_global = value_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim
        )
        value_vectors_only_global[is_local_index_global_attn_nonzero] = value_vectors[is_index_global_attn_nonzero]

        # use `matmul` because `einsum` crashes sometimes with fp16
        # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
        # compute attn output only global
        attn_output_only_global = torch.matmul(
            attn_probs_only_global.transpose(1, 2).clone(
            ), value_vectors_only_global.transpose(1, 2).clone()
        ).transpose(1, 2)

        # reshape attn probs
        attn_probs_without_global = attn_probs.narrow(
            -1, max_num_global_attn_indices, attn_probs.size(-1) - max_num_global_attn_indices
        ).contiguous()

        # compute attn output with global
        attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(
            attn_probs_without_global, value_vectors, self.one_sided_attn_window_size
        )
        return attn_output_only_global + attn_output_without_global

    def _compute_global_attn_output_from_hidden(
        self,
        hidden_states,
        context_embedded,
        max_num_global_attn_indices,
        layer_head_mask,
        is_local_index_global_attn_nonzero,
        is_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
        is_index_masked,
    ):
        seq_len, batch_size = hidden_states.shape[:2]

        # prepare global hidden states
        global_attn_hidden_states = hidden_states.new_zeros(
            max_num_global_attn_indices, batch_size, self.embed_dim)
        global_attn_hidden_states[is_local_index_global_attn_nonzero[::-1]
                                  ] = hidden_states[is_index_global_attn_nonzero[::-1]]

        # prepare global context information
        global_attn_context_embedded = context_embedded.new_zeros(
            max_num_global_attn_indices, batch_size, self.embed_dim)
        global_attn_context_embedded[is_local_index_global_attn_nonzero[::-1]
                                     ] = context_embedded[is_index_global_attn_nonzero[::-1]]

        # global key, query, value
        global_query_vectors_only_global = self.query_global(
            global_attn_hidden_states)
        global_key_vectors = self.key_global(hidden_states)
        global_value_vectors = self.value_global(hidden_states)

        # global context_key, context_query
        global_context_query_vectors_only_global = self.context_for_q(
            global_attn_context_embedded)
        global_context_key_vectors = self.context_for_k(context_embedded)

        # normalize
        global_query_vectors_only_global /= math.sqrt(self.head_dim)
        global_context_query_vectors_only_global /= math.sqrt(self.head_dim)

        # reshape
        global_query_vectors_only_global = (
            global_query_vectors_only_global.contiguous()
            .view(max_num_global_attn_indices, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )  # (batch_size * self.num_heads, max_num_global_attn_indices, head_dim)
        global_key_vectors = (
            global_key_vectors.contiguous().view(-1, batch_size * self.num_heads,
                                                 self.head_dim).transpose(0, 1)
        )  # batch_size * self.num_heads, seq_len, head_dim)
        global_value_vectors = (
            global_value_vectors.contiguous().view(-1, batch_size * self.num_heads,
                                                   self.head_dim).transpose(0, 1)
        )  # batch_size * self.num_heads, seq_len, head_dim)
        global_context_query_vectors_only_global = (
            global_context_query_vectors_only_global.contiguous()
            .view(max_num_global_attn_indices, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )  # (batch_size * self.num_heads, max_num_global_attn_indices, head_dim)
        global_context_key_vectors = (
            global_context_key_vectors.contiguous().view(-1, batch_size * self.num_heads,
                                                         self.head_dim).transpose(0, 1)
        )  # batch_size * self.num_heads, seq_len, head_dim)

        # compute attn scores
        global_attn_scores = torch.bmm(
            global_query_vectors_only_global, global_key_vectors.transpose(1, 2))
        global_quasi_attn_scores = torch.bmm(
            global_context_query_vectors_only_global, global_context_key_vectors.transpose(1, 2))

        global_attn_scores = global_attn_scores.view(
            batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
        global_quasi_attn_scores = global_quasi_attn_scores.view(
            batch_size, self.num_heads, max_num_global_attn_indices, seq_len)

        global_attn_scores[
            is_local_index_no_global_attn_nonzero[0], :, is_local_index_no_global_attn_nonzero[1], :
        ] = -10000.0
        global_attn_scores = global_attn_scores.masked_fill(
            is_index_masked[:, None, None, :],
            -10000.0,
        )

        global_quasi_attn_scores[
            is_local_index_no_global_attn_nonzero[0], :, is_local_index_no_global_attn_nonzero[1], :
        ] = -10000.0
        global_quasi_attn_scores = global_quasi_attn_scores.masked_fill(
            is_index_masked[:, None, None, :],
            -10000.0,
        )

        global_attn_scores = global_attn_scores.view(
            batch_size * self.num_heads, max_num_global_attn_indices, seq_len)
        global_quasi_attn_scores = global_quasi_attn_scores.view(
            batch_size * self.num_heads, max_num_global_attn_indices, seq_len)

        # compute global attn probs
        global_attn_probs_float = nn.functional.softmax(
            global_attn_scores, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability
        
        quasi_scalar = 1.0
        quasi_attention_scores = 1.0 * quasi_scalar * \
            self.quasi_act(global_quasi_attn_scores)
        lambda_q_context = self.lambda_q_context_layer(global_context_query_vectors_only_global*math.sqrt(self.head_dim))
        lambda_q_query = self.lambda_q_query_layer(global_query_vectors_only_global*math.sqrt(self.head_dim))
        lambda_q = self.quasi_act(lambda_q_context + lambda_q_query)
        lambda_k_context = self.lambda_k_context_layer(global_context_key_vectors)
        lambda_k_key = self.lambda_k_key_layer(global_key_vectors)
        lambda_k = self.quasi_act(lambda_k_context + lambda_k_key)
        lambda_q_scalar = 1.0
        lambda_k_scalar = 1.0
        lambda_context = lambda_q_scalar*lambda_q + lambda_k_scalar*lambda_k
        lambda_context = (1 - lambda_context)
        global_quasi_attn_prob = lambda_context * quasi_attention_scores
        global_quasi_attn_prob = global_quasi_attn_prob[:, :1, :]
        new_attn_probs = global_attn_probs_float + global_quasi_attn_prob

        # apply layer head masking
        if layer_head_mask is not None:
            assert layer_head_mask.size() == (
                self.num_heads,
            ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
            new_attn_probs = layer_head_mask.view(1, -1, 1, 1) * new_attn_probs.view(
                batch_size, self.num_heads, max_num_global_attn_indices, seq_len
            )
            new_attn_probs = new_attn_probs.view(
                batch_size * self.num_heads, max_num_global_attn_indices, seq_len
            )

        new_attn_probs = nn.functional.dropout(
            new_attn_probs.type_as(global_attn_scores), p=self.dropout, training=self.training
        )
        del global_attn_scores
        del global_attn_probs_float
        del global_quasi_attn_prob

        # global attn output
        global_attn_output = torch.bmm(new_attn_probs, global_value_vectors)

        assert list(global_attn_output.size()) == [
            batch_size * self.num_heads,
            max_num_global_attn_indices,
            self.head_dim,
        ], f"global_attn_output tensor has the wrong size. Size should be {(batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim)}, but is {global_attn_output.size()}."

        new_attn_probs = new_attn_probs.view(
            batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
        global_attn_output = global_attn_output.view(
            batch_size, self.num_heads, max_num_global_attn_indices, self.head_dim
        )
        return global_attn_output, new_attn_probs

class BERTSelfOutput1(nn.Module):
    def __init__(self, config):
        super(BERTSelfOutput1, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                vocab_size=32000,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=16,
                initializer_range=0.02,
                full_pooler=False): # this is for transformer-like BERT
        """Constructs BertConfig.
        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.full_pooler = full_pooler

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class BERTLayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    Args:
        x: torch.Tensor x:
    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx


class BERTEmbeddings1(nn.Module):
    def __init__(self, config):
        super(BERTEmbeddings1, self).__init__()
        """Construct the embedding module from word, position and token_type embeddings.
        """
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(
            config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute")

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx).to(input_ids.device)
                
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
            
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class ContextBERTPooler1(nn.Module):
    def __init__(self, config):
        super(ContextBERTPooler1, self).__init__()
        # old fileds
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states, attention_mask):
        hs_pooled = hidden_states[:, 0]
        #######################################################################

        #return first_token_tensor
        pooled_output = self.dense(hs_pooled)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BERTOutput1(nn.Module):
    def __init__(self, config):
        super(BERTOutput1, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class ContextBERTAttention1(nn.Module):
    def __init__(self, config, layer_id=0):
        super(ContextBERTAttention1, self).__init__()
        self.self = ContextBERTSelfAttention1(config, layer_id)
        self.output = BERTSelfOutput1(config)

    def forward(self, input_tensor, attention_mask,
                # optional parameters for saving context information
                device=None, 
                layer_head_mask=None,
                is_index_masked=None,
                is_index_global_attn=None,
                is_global_attn=None,
                context_embedded=None):
        self_output, new_attention_probs = \
            self.self.forward(input_tensor, attention_mask, device,
                            layer_head_mask=layer_head_mask,
                           is_index_masked=is_index_masked,
                           is_index_global_attn=is_index_global_attn,
                           is_global_attn=is_global_attn,
                           context_embedded=context_embedded,)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, new_attention_probs

def apply_chunking_to_forward(
    forward_fn, chunk_size: int, chunk_dim: int, *input_tensors
) -> torch.Tensor:
    """
    This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension
    `chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.
    If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly
    applying `forward_fn` to `input_tensors`.
    Args:
        forward_fn (`Callable[..., torch.Tensor]`):
            The forward function of the model.
        chunk_size (`int`):
            The chunk size of a chunked tensor: `num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (`int`):
            The dimension over which the `input_tensors` should be chunked.
        input_tensors (`Tuple[torch.Tensor]`):
            The input tensors of `forward_fn` which will be chunked
    Returns:
        `torch.Tensor`: A tensor with the same shape as the `forward_fn` would have given if applied`.
    Examples:
    ```python
    # rename the usual forward() fn to forward_chunk()
    def forward_chunk(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states
    # implement a chunked forward function
    def forward(self, hidden_states):
        return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    ```"""

    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
            "tensors are given"
        )

    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}"
                )

        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)

class ContextBERTLayer1(nn.Module):
    def __init__(self, config, layer_id=0):
        super(ContextBERTLayer1, self).__init__()
        self.attention = ContextBERTAttention1(config, layer_id)
        self.intermediate = BERTIntermediate1(config)
        self.output = BERTOutput1(config)
        self.chunk_size_feed_forward = config.attention_window[0]
        self.seq_len_dim = 1
        self.config = config

    def forward(self, hidden_states, attention_mask,
                # optional parameters for saving context information
                device=None, 
                layer_head_mask=None,
                is_index_masked=None,
                is_index_global_attn=None,
                is_global_attn=None,
                context_embedded=None):
        attention_output, new_attention_probs = \
            self.attention(hidden_states, attention_mask, device, 
                           layer_head_mask=layer_head_mask,
                           is_index_masked=is_index_masked,
                           is_index_global_attn=is_index_global_attn,
                           is_global_attn=is_global_attn,
                           context_embedded=context_embedded,)
        
        #######################################################################
        #######################################################################
        # Apply chunking to feed-forward layer to save memory
        layer_output = apply_chunking_to_forward(
            self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        #######################################################################
        #######################################################################
        
        return layer_output, new_attention_probs, self.config, context_embedded
    
    def ff_chunk(self, attn_output):
        intermediate_output = self.intermediate(attn_output)
        layer_output = self.output(intermediate_output, attn_output)
        return layer_output

class ContextBERTEncoder1(nn.Module):
    def __init__(self, config):
        super(ContextBERTEncoder1, self).__init__()

        deep_context_transform_layer = \
            nn.Linear(2*config.hidden_size, config.hidden_size)
        self.context_layer = \
            nn.ModuleList([copy.deepcopy(deep_context_transform_layer) for _ in range(config.num_hidden_layers)])  

        self.layer = nn.ModuleList([copy.deepcopy(ContextBERTLayer1(config, layer_id=_)) for _ in range(config.num_hidden_layers)])    

    def forward(self, hidden_states, attention_mask,
                # optional parameters for saving context information
                device=None, context_embeddings=None,
                head_mask = None,
                padding_len=0,):
        #######################################################################
        # Here, we can try other ways to incoperate the context!
        # Just make sure the output context_embedded is in the 
        # shape of (batch_size, seq_len, d_hidden).

        #######################################################################
        #######################################################################
        # Identify global attention indices
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()
        #######################################################################
        #######################################################################
        
        all_encoder_layers = []
        all_new_attention_probs = []
        
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layer)
            ), f"The head_mask should be specified for {len(self.layer)} layers, but it is for {head_mask.size()[0]}."
            
        for idx, layer_module in enumerate(self.layer):
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            all_encoder_layers.append(hidden_states)
            # update context
            deep_context_hidden = torch.cat([context_embeddings, hidden_states], dim=-1)
            deep_context_hidden = self.context_layer[idx](deep_context_hidden)
            deep_context_hidden += context_embeddings
            # BERT encoding
            hidden_states, new_attention_probs, config1, ce = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer_module), hidden_states, attention_mask, device,
                head_mask[idx] if head_mask is not None else None,
                is_index_masked,
                is_index_global_attn,
                is_global_attn,
                deep_context_hidden,
                )
            all_new_attention_probs.append(new_attention_probs.clone())
            
        all_encoder_layers.append(hidden_states)

        #######################################################################
        #######################################################################
        # undo padding
        if padding_len > 0:
            all_encoder_layers = [state[:, :-padding_len] for state in all_encoder_layers]
            all_new_attention_probs = [state[:, :-padding_len, :, :] for state in all_new_attention_probs]
        #######################################################################
        #######################################################################

        return all_encoder_layers, all_new_attention_probs, config1, ce

def get_extended_attention_mask(attention_mask, input_shape, device):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device: (`torch.device`):
                The device of the input to the model.
        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=torch.long)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

class ContextBertModel1(nn.Module):
    """ Context-aware BERT base model
    """
    def __init__(self, config):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(ContextBertModel1, self).__init__()
        config.attention_window = [512] * config.num_hidden_layers
        config.pad_token_id = 1
        self.embeddings = BERTEmbeddings1(config)
        self.encoder = ContextBERTEncoder1(config)
        self.pooler = ContextBERTPooler1(config)

        self.context_embeddings = nn.Embedding(2375, config.hidden_size)
        self.config = config
        
    def _pad_to_window_size(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        pad_token_id: int,
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        # padding
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )

        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]

        padding_len = (attention_window - seq_len %
                       attention_window) % attention_window
        if padding_len > 0:
            logger.info(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.attention_window`: {attention_window}"
            )
            if input_ids is not None:
                input_ids = nn.functional.pad(
                    input_ids, (0, padding_len), value=pad_token_id)
            if position_ids is not None:
                # pad with position_id = pad_token_id as in modeling_roberta.RobertaEmbeddings
                position_ids = nn.functional.pad(
                    position_ids, (0, padding_len), value=pad_token_id)

            attention_mask = nn.functional.pad(
                attention_mask, (0, padding_len), value=False
            )  # no attention on the padding tokens
            token_type_ids = nn.functional.pad(
                token_type_ids, (0, padding_len), value=0)  # pad with token_type_id = 0

        return padding_len, input_ids, attention_mask, token_type_ids, position_ids
        
    def _merge_to_attention_mask(self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor):
        # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            # simply use `global_attention_mask` as `attention_mask`
            # if no `attention_mask` is given
            attention_mask = global_attention_mask + 1
        return attention_mask

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                # optional parameters for saving context information
                device=None, context_ids=None,
                global_attention_mask = None,
                head_mask = None,
                position_ids = None,):
        input_shape = input_ids.size()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        #######################################################################
        #######################################################################
        # Merge local attention mask and global attention mask
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)
        
        # Pad according to window size
        padding_len, input_ids, attention_mask, token_type_ids, position_ids = self._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            pad_token_id=self.config.pad_token_id,
        )

        # Utilize -10000 to realize removing effect
        extended_attention_mask = get_extended_attention_mask(attention_mask, input_shape, device)[
            :, 0, 0, :
        ]
        #######################################################################
        #######################################################################

        embedding_output = self.embeddings(input_ids, token_type_ids)

        #######################################################################
        # Context embeddings
        seq_len = embedding_output.shape[1]
        context_embedded = self.context_embeddings(context_ids).squeeze(dim=1)
        context_embedding_output = torch.stack(seq_len*[context_embedded], dim=1)
        #######################################################################
        
        all_encoder_layers, all_new_attention_probs, config1, ce = \
            self.encoder(embedding_output, extended_attention_mask, device,
                        context_embedding_output,
                        head_mask=head_mask,
                        padding_len=padding_len)
        sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(sequence_output, attention_mask)
        return pooled_output, all_encoder_layers, all_new_attention_probs, config1, ce

class QACGBertForSequenceClassification1(nn.Module):
    """Proposed Context-Aware Bert Model for Sequence Classification
    """
    def __init__(self, config, num_labels, init_weight=False, init_lrp=False):
        super(QACGBertForSequenceClassification1, self).__init__()
        self.bert = ContextBertModel1(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.num_head = config.num_attention_heads
        self.config = config
        if init_weight:
            print("init_weight = True")
            def init_weights(module):
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    # Slightly different from the TF version which uses truncated_normal for initialization
                    # cf https://github.com/pytorch/pytorch/pull/5617
                    module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                elif isinstance(module, BERTLayerNorm):
                    module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                    module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        module.bias.data.zero_()
            self.apply(init_weights)

        #######################################################################
        # Let's do special handling of initialization of newly added layers
        # for newly added layer, we want it to be "invisible" to the
        # training process in the beginning and slightly diverge from the
        # original model. To illustrate, let's image we add a linear layer
        # in the middle of a pretrain network. If we initialize the weight
        # randomly by default, then it will effect the output of the 
        # pretrain model largely so that it lost the point of importing
        # pretrained weights.
        #
        # What we do instead is that we initialize the weights to be a close
        # diagonal identity matrix, so that, at the beginning, for the 
        # network, it will be bascially copying the input hidden vectors as
        # the output, and slowly diverge in the process of fine tunning. 
        # We turn the bias off to accomedate this as well.
        init_perturbation = 1e-2
        for layer_module in self.bert.encoder.layer:
            layer_module.attention.self.lambda_q_context_layer.weight.data.normal_(mean=0.0, std=init_perturbation)
            layer_module.attention.self.lambda_k_context_layer.weight.data.normal_(mean=0.0, std=init_perturbation)
            layer_module.attention.self.lambda_q_query_layer.weight.data.normal_(mean=0.0, std=init_perturbation)
            layer_module.attention.self.lambda_k_key_layer.weight.data.normal_(mean=0.0, std=init_perturbation)

        #######################################################################
        if init_lrp:
            print("init_lrp = True")
            init_hooks_lrp(self)

    def forward(self, input_ids, token_type_ids, attention_mask, seq_lens,
                device=None, labels=None,
                # optional parameters for saving context information
                context_ids=None,
                global_attention_mask = None,
                head_mask = None,
                position_ids = None,
                ):
        #######################################################################
        #######################################################################
        # Global attention initialization
        if global_attention_mask is None:
            logger.info("Initializing global attention on CLS token...")
            global_attention_mask = torch.zeros_like(input_ids)
            # global attention on cls token
            global_attention_mask[:, 0] = 1
        #######################################################################
        #######################################################################

        pooled_output, all_encoder_layers, all_new_attention_probs, config1, ce = \
            self.bert(input_ids, token_type_ids, attention_mask, device, context_ids, 
                        global_attention_mask=global_attention_mask,
                        head_mask=head_mask,
                        position_ids=position_ids,)
        
        pooled_output = self.dropout(pooled_output)

        logits = \
            self.classifier(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits, pooled_output, all_encoder_layers, all_new_attention_probs, config1
        else:
            return logits

    def backward_gradient(self, sensitivity_grads):
        classifier_out = func_activations['model.classifier']
        embedding_output = func_activations['model.bert.embeddings']
        sensitivity_grads = torch.autograd.grad(classifier_out, embedding_output, 
                                                grad_outputs=sensitivity_grads)[0]
        return sensitivity_grads

