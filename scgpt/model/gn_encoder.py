from torch import nn
from transformers.modeling_utils import apply_chunking_to_forward
from .gn_attn import gn_attn
from .gn_utils import gn_ffn, gn_output

class gn_transformer_encoder(nn.Module):
    def __init__(self, encoder_layer, config, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer(config, i) for i in range(num_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        g= None,
        attention_mask=None,
        head_mask=None,
        padding_len=0,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):

        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None  # All local attentions.
        all_global_attentions = () if (output_attentions and is_global_attn) else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, is_global_attn, output_attentions)

                    return custom_forward

                from torch.utils.checkpoint import checkpoint
                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    g,
                    attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    is_index_masked,
                    is_index_global_attn,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    g=g,
                    attention_mask=attention_mask,
                    layer_head_mask=head_mask[idx] if head_mask is not None else None,
                    is_index_masked=is_index_masked,
                    is_index_global_attn=is_index_global_attn,
                    is_global_attn=is_global_attn,
                    output_attentions=output_attentions,
                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                # bzs x seq_len x num_attn_heads x (num_global_attn + attention_window_len + 1) => bzs x num_attn_heads x seq_len x (num_global_attn + attention_window_len + 1)
                all_attentions = all_attentions + (layer_outputs[1].transpose(1, 2),)

                if is_global_attn:
                    # bzs x num_attn_heads x num_global_attn x seq_len => bzs x num_attn_heads x seq_len x num_global_attn
                    all_global_attentions = all_global_attentions + (layer_outputs[2].transpose(2, 3),)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # undo padding
        if padding_len > 0:
            # unpad `hidden_states` because the calling function is expecting a length == input_ids.size(1)
            hidden_states = hidden_states[:, :-padding_len]
            if output_hidden_states:
                all_hidden_states = tuple([state[:, :-padding_len] for state in all_hidden_states])

            if output_attentions:
                all_attentions = tuple([state[:, :, :-padding_len, :] for state in all_attentions])

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_global_attentions] if v is not None
            )
        return hidden_states


class gn_layer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.attention = gn_attn(config, layer_id)
        self.intermediate = gn_ffn(config)
        self.output = gn_output(config)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

    def forward(
            self,
            hidden_states,
            g=None,
            attention_mask=None,
            layer_head_mask=None,
            is_index_masked=None,
            is_index_global_attn=None,
            is_global_attn=None,
            output_attentions=False,
    ):
        self_attn_outputs = self.attention(
            hidden_states,
            g=g,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )

        attn_output = self_attn_outputs[0]
        outputs = self_attn_outputs[1:]

        layer_output = apply_chunking_to_forward(
            self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attn_output
        )
        outputs = (layer_output,) + outputs
        return outputs

    def ff_chunk(self, attn_output):
        intermediate_output = self.intermediate(attn_output)
        layer_output = self.output(intermediate_output, attn_output)
        return layer_output

if __name__ == "__main__":
    from types import SimpleNamespace
    class dict_2_class(SimpleNamespace):
        pass
    config = {
        "hidden_size": 512,
        "num_attention_heads": 8,
        'hidden_dropout_prob': 0.2,
        "attention_probs_dropout_prob": 0.2,
        'layer_norm_eps':1e-5, # 1e-5 layer_norm_eps
        'intermediate_size': 2048, # d_hid
        'hidden_act': 'relu',# activation
        'chunk_size_feed_forward':0,
    }
    config = dict_2_class(**config)
    transformer_encoder = gn_transformer_encoder(gn_layer, config, 12)
    print('Test pass!')