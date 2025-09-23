"""
test sg2260e structure
"""

from torch_tpu.utils.reflection.time_model import *

import random


def test_in_core(core_id=0):
    # test multi engin in one core
    tiu = TiuSpan.from_duration(random.random() * 5)
    gdma = GdmaSpan.from_duration(random.random() * 500)

    pipeline = Pipeline()
    pipeline.add_child(TiuSpan.from_duration(random.random() * 5))
    pipeline.add_child(GdmaSpan.from_duration(random.random() * 5))
    pipeline.add_child(TiuSpan.from_duration(random.random() * 5))

    core = Core(core_id=core_id).add_child(pipeline, tiu, gdma)
    return core


def test_in_chip(chip_id=0):
    chip = Chip(chip_id=chip_id)
    for i in range(8):
        core = test_in_core(i)
        chip.add_child(core)
        chip.add_child(CdmaSpan.from_duration(random.random() * 5))

    return chip


def test_in_card(card_id=0):
    card = Card(card_id=card_id)
    for i in range(2):
        chip = test_in_chip(i)
        card.add_child(chip)

    return card


def test_in_panel(panel_id=0):
    panel = Panel(panel_id=panel_id)
    for i in range(4):
        card = test_in_card(i)
        panel.add_child(card)

    return panel


def test_cluster():
    cluster = Cluster()
    for i in range(4):
        panel = test_in_panel(i)
        cluster.add_child(panel)

    return cluster
