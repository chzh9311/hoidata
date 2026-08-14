"""ARCTIC dataset constants.

Source: Text2HOI (https://github.com/JunukCha/Text2HOI.git)
  constants/arctic_constants.py  (object/action/verb lists)

ARCTIC objects are *articulated* (a top + bottom part joined by a 1-DoF revolute
joint).  The object x-data is 10-D: trans(3) + rot6d(6) + articulation_angle(1),
where rot6d is pytorch3d's convention (first two *rows* of R) — see
``hoidata.utils.proc.transform_obj_to_xdata``.

Splits (val/test object held-out sets) do not exist upstream; the lists below are
chosen here so the action-classify / evaluate tasks have disjoint object splits —
tune as needed.  The generation task uses every object.
"""

arctic_obj_name = [
    "box", "capsulemachine", "ketchup", "laptop", "microwave",
    "mixer", "notebook", "phone", "scissors", "waffleiron",
]

# Held-out object splits (chosen here — not from the original ARCTIC release).
arctic_val_obj_name = [
    "ketchup",
]

arctic_test_obj_name = [
    "microwave", "notebook",
]

subj_list = [
    "s01", "s02", "s04", "s05",
    "s06", "s07", "s08", "s09",
    "s10",
]

# (verb action_name) stored per-sample; process_text combines "{verb} {object}".
action_list = [
    "close", "cut", "grasp", "lift", "open",
    "operate", "place", "point", "shake", "sprinkle", "type",
]

# Excluded "{verb} {object}" pairs in the raw ARCTIC annotation files.
# These clips are skipped during preprocessing.
not_save_list = [
    "write notebook", "operate waffleiron",
    "push capsulemachine",
]

# Full "{verb} {object}" action list used by preprocessing_text() to build the
# text-description JSON.  Derived from the ARCTIC annotation corpus.
action_verb_obj_list = [
    'close box', 'close capsulemachine', 'close ketchup',
    'close laptop', 'close microwave', 'close mixer', 'close notebook',
    'close phone', 'close waffleiron', 'cut scissors', 'grasp box',
    'grasp capsulemachine', 'grasp ketchup', 'grasp laptop',
    'grasp microwave', 'grasp mixer', 'grasp notebook', 'grasp phone',
    'grasp scissors', 'grasp waffleiron', 'lift box',
    'lift capsulemachine', 'lift ketchup', 'lift laptop',
    'lift microwave', 'lift mixer', 'lift notebook', 'lift phone',
    'lift scissors', 'lift waffleiron', 'open box',
    'open capsulemachine', 'open ketchup', 'open laptop',
    'open microwave', 'open mixer', 'open notebook', 'open phone',
    'open waffleiron', 'operate microwave', 'place box',
    'place capsulemachine', 'place ketchup', 'place laptop',
    'place microwave', 'place mixer', 'place notebook', 'place phone',
    'place scissors', 'place waffleiron', 'point notebook',
    'shake ketchup', 'sprinkle ketchup', 'type laptop', 'type phone',
]

present_participle = {
    "close": "closing",
    "cut": "cutting",
    "grasp": "grasping",
    "lift": "lifting",
    "open": "opening",
    "operate": "operating",
    "place": "placing",
    "point": "pointing",
    "shake": "shaking",
    "sprinkle": "sprinkling",
    "type": "typing",
}

third_verb = {
    "close": "closes",
    "cut": "cuts",
    "grasp": "grasps",
    "lift": "lifts",
    "open": "opens",
    "operate": "operates",
    "place": "places",
    "point": "points",
    "shake": "shakes",
    "sprinkle": "sprinkles",
    "type": "types",
}

passive_verb = {
    "close": "is closed",
    "cut": "is cut",
    "grasp": "is grasped",
    "lift": "is lifted",
    "open": "is opened",
    "operate": "is operated",
    "place": "is placed",
    "point": "is pointed",
    "shake": "is shaken",
    "sprinkle": "is sprinkled",
    "type": "is typed",
}
