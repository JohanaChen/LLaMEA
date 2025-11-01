from .state import _CHOICE_STATE

def reset_choice_state():
    _CHOICE_STATE.update({
        "incumbent_id": None,        # current champion's id (A)
        "incumbent_render": None,    # pretty text for A
        "incumbent_json": None,      # parsed JSON for A
        "fitness_level": 1.0,        # baseline to compare against
        "printed_help": False,       # show instructions once
        "last_feedback": None,       # último feedback escrito por el usuario
        "incumbent_feedback": None,  # feedback asociado al incumbente actual
    })
    