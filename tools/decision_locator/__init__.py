"""decision-locator: find and steer the layer where an agent commits a tool-calling decision.

See README.md and paper #6 ("The Lever Is Late"). Quick start:

    from decision_locator import DecisionLocator
    loc = DecisionLocator(model, tokenizer)
    gaps  = loc.locate(ids, target_id, alt_ids)            # where is the decision readable?
    donor = loc.donor_state([success_ids], layer=55)       # a (task-matched) donor state
    dP    = loc.sweep_patch(ids, {55: donor}, [55], target_id, alt_ids)   # where is it writable?
    text  = loc.steer_generate(ids, layer=55, donor=donor) # does it emit a real action?
"""
from .locator import DecisionLocator, commitment_layer

__all__ = ["DecisionLocator", "commitment_layer"]
__version__ = "0.1.0"
