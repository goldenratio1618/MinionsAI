from minionsai.discriminator_only.translator import MAX_SPEED_OR_RANGE


class UnitType():
    def __init__(self, name, cost, rebate, attack, defense, speed=1, attack_range=1, persistent=False, immune=False, max_stack=1, spawn=False, blink=False, unsummoner=False, deadly=False, flurry=False, flying=False, lumbering=False, terrain_ability=0):
        self.name = name
        self.attack = attack
        self.defense = defense
        self.speed = speed
        self.attack_range = attack_range
        self.persistent = persistent
        self.immune = immune
        self.max_stack = max_stack
        self.spawn = spawn
        self.blink = blink
        self.unsummoner = unsummoner
        self.deadly = deadly
        if unsummoner or deadly:
            self.attack = 1
        self.flurry = flurry
        self.flying = flying
        self.lumbering = lumbering
        self.terrain_ability = terrain_ability
        self.cost = cost
        self.rebate = rebate

    def __eq__(self, __o: object) -> bool:
        return self.name == __o.name

NECROMANCER = UnitType("Necromancer", 255, 0, 0, 7, persistent=True, immune=True, spawn=True, unsummoner=True)
ZOMBIE = UnitType("Zombie", 2, 0, 1, 2, lumbering=True)
MAX_SPEED_OR_RANGE = 1

unitList = [NECROMANCER, ZOMBIE]

def unit_type_from_name(unit_name):
    for unit in unitList:
        if unit.name.lower() == unit_name.lower():
            return unit
    return None

def flexible_unit_type(unit_type_str: str):
    if unit_type_str.isdigit():
        return unitList[int(unit_type_str)]
    elif len(unit_type_str) == 1:
        return next(unit for unit in unitList if unit.name[0].lower() == unit_type_str.lower())
    else:
        return unit_type_from_name(unit_type_str)
