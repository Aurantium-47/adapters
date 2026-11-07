import contextlib
import copy
import unittest

import torch

import adapters
from adapters import AdapterSetup, LoRAConfig, SeqBnConfig, Stack
from transformers import BertConfig, BertForSequenceClassification


class AdapterGradientCheckpointingTest(unittest.TestCase):
    def test_checkpointing_preserves_adapter_setup(self):
        for adapter_config in (SeqBnConfig(reduction_factor=2), LoRAConfig(r=2)):
            for forward_setup in (None, "a", Stack("a", "b")):
                for backward_setup in (None, "b"):
                    with self.subTest(config=adapter_config, forward=forward_setup, backward=backward_setup):
                        torch.manual_seed(42)
                        model = BertForSequenceClassification(
                            BertConfig(
                                vocab_size=30,
                                hidden_size=16,
                                num_hidden_layers=2,
                                num_attention_heads=2,
                                intermediate_size=32,
                                hidden_dropout_prob=0.0,
                                attention_probs_dropout_prob=0.0,
                            )
                        )
                        adapters.init(model)
                        for name in ("a", "b"):
                            model.add_adapter(name, config=adapter_config)
                        model.train_adapter(Stack("a", "b"))
                        model.set_active_adapters("b")
                        reference = copy.deepcopy(model)
                        model.gradient_checkpointing_enable()
                        inputs = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]), "labels": torch.tensor([0, 1])}
                        for current in (reference, model):
                            forward_context = (
                                AdapterSetup(forward_setup) if forward_setup else contextlib.nullcontext()
                            )
                            with forward_context:
                                loss = current(**inputs).loss
                            backward_context = (
                                AdapterSetup(backward_setup) if backward_setup else contextlib.nullcontext()
                            )
                            with backward_context:
                                loss.backward(retain_graph=True)
                                # Checkpoint recomputation contexts must also support a retained graph.
                                loss.backward()
                                self.assertEqual(
                                    AdapterSetup.get_context_adapter_setup(),
                                    Stack(backward_setup) if backward_setup else None,
                                )
                            self.assertIsNone(AdapterSetup.get_context())
                        for (name, parameter), (_, expected) in zip(
                            model.named_parameters(), reference.named_parameters()
                        ):
                            if expected.grad is None:
                                self.assertIsNone(parameter.grad, name)
                            else:
                                self.assertIsNotNone(parameter.grad, name)
                                torch.testing.assert_close(
                                    parameter.grad, expected.grad, rtol=1e-5, atol=1e-9, msg=name
                                )
