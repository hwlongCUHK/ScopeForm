"""SSV paper-method reconstruction for CPM's public, multi-claim task.

Source: Raza & Milic-Frayling, IJCAI 2025, arXiv:2501.16961v3, Figure 5
and Appendix 7. The linked official repository was empty on 2026-11-12.
No evaluator targets or evidence-note components are used here.
"""
from __future__ import annotations

import json
from typing import Any

TEMPERATURES = (0.0, 0.3, 0.4, 0.5)
MAX_REPAIRS = 2
LABELS = {"supported", "refuted", "not_determined"}

# Published Appendix 7 prompts and all demonstrations. CC BY 4.0; see ssv_METHOD.md.
PUBLISHED_PROMPTS = {'decomposition': 'Given a problem description, please decompose it into an initial context and a list of independent constraints. If there is no explicit initial context given and only constraints are given, then just state "None" for initial context. Some examples are given below. \n\n------ \n\nProblem: \n\nThe bald eagle eats the cow. The bald eagle is red. The bald eagle needs the cow. The bear needs the rabbit. The cow is kind. The cow is red. The cow needs the bald eagle. The rabbit eats the bear. The rabbit eats the cow. The rabbit sees the cow. If something needs the bald eagle then it needs the rabbit. If the bald eagle is nice and the bald eagle is young then the bald eagle sees the cow. If the rabbit needs the cow then the cow sees the rabbit. If something eats the cow and the cow is nice then it needs the bald eagle. If something needs the rabbit then it is nice. If something sees the rabbit then it is red. If something needs the bald eagle then it eats the bald eagle. \n\nInitialContext: \n\nNone \n\nConstraints: \n\nThe bald eagle eats the cow. \n\n### \n\nThe bald eagle is red. \n\n### \n\nThe bald eagle needs the cow. \n\n### \n\nThe bear needs the rabbit. \n\n### \n\nThe cow is kind. \n\n### \n\nThe cow is red. \n\n### \n\nThe cow needs the bald eagle. \n\n### \n\nThe rabbit eats the bear. \n\n### \n\nThe rabbit eats the cow. \n\n### \n\nThe rabbit sees the cow. \n\n### \n\nIf something needs the bald eagle then it needs the rabbit. \n\n### \n\nIf the bald eagle is nice and the bald eagle is young then the bald eagle sees the cow. \n\n### \n\nIf the rabbit needs the cow then the cow sees the rabbit. \n\n### \n\nIf something eats the cow and the cow is nice then it needs the bald eagle. \n\n### \n\nIf something needs the rabbit then it is nice. \n\n### \n\nIf something sees the rabbit then it is red. \n\n### \n\nIf something needs the bald eagle then it eats the bald eagle. \n\n------ \n\nProblem: \n\nOn Tuesday Vladimir and Wendy each eat exactly four separate meals: breakfast, lunch, dinner, and a snack. The following is all that is known about what they eat during that day: At no meal does Vladimir eat the same kind of food as Wendy. Neither of them eats the same kind of food more than once during the day. For breakfast, each eats exactly one of the following: hot cakes, poached eggs, or omelet. For lunch, each eats exactly one of the following: fish, hot cakes, macaroni, or omelet. For dinner, each eats exactly one of the following: fish, hot cakes, macaroni, or omelet. For a snack, each eats exactly one of the following: fish or omelet. Wendy eats an omelet for lunch. \n\nInitialContext: \n\nOn Tuesday Vladimir and Wendy each eat exactly four separate meals: breakfast, lunch, dinner, and a snack. \n\nConstraints: \n\nAt no meal does Vladimir eat the same kind of food as Wendy. \n\n### \n\nNeither of them eats the same kind of food more than once during the day. \n\n### \n\nFor breakfast, each eats exactly one of the following: hot cakes, poached eggs, or omelet. \n\n### \n\nFor lunch, each eats exactly one of the following: fish, hot cakes, macaroni, or omelet. \n\n### \n\nFor dinner, each eats exactly one of the following: fish, hot cakes, macaroni, or omelet. \n\n### \n\nFor a snack, each eats exactly one of the following: fish or omelet. \n\n### \n\nWendy eats an omelet for lunch. \n\n------ \n\nProblem: \n\nIn a repair facility there are exactly six technicians: Stacy, Urma, Wim, Xena, Yolanda, and Zane. Each technician repairs machines of at least one of the following three types-radios, televisions, and VCRs-and no other types. The following conditions apply: Xena and exactly three other technicians repair radios. Yolanda repairs both televisions and VCRs. Stacy does not repair any type of machine that Yolanda repairs. Zane repairs more types of machines than Yolanda repairs. Wim does not repair any type of machine that Stacy repairs. Urma repairs exactly two types of machines. \n\nInitialContext: \n\nIn a repair facility there are exactly six technicians: Stacy, Urma, Wim, Xena, Yolanda, and Zane. Each technician repairs machines of at least one of the following three types-radios, televisions, and VCRs-and no other types. \n\nConstraints: \n\nXena and exactly three other technicians repair radios. \n\n### \n\nYolanda repairs both televisions and VCRs. \n\n### \n\nStacy does not repair any type of machine that Yolanda repairs. \n\n### \n\nZane repairs more types of machines than Yolanda repairs. \n\n### \n\nWim does not repair any type of machine that Stacy repairs. \n\n### \n\nUrma repairs exactly two types of machines. \n\n------', 'incremental': 'Given a z3 program that models a particular problem and a new constraint described in natural language, please provide the z3 code to augment the program with the new constraint. Please provide only the z3 program code in the output and no other markdown formatting or explanatory text. \n\n------ \n\nExistingProgram: \n\n# On Tuesday Vladimir and Wendy each eat exactly four separate meals: breakfast, lunch, dinner, and a snack. \n\nfrom z3 import * \n\npeople_sort, (Vladimir, Wendy) = EnumSort(’people’, [’Vladimir’, ’Wendy’]) \n\nmeals_sort, (breakfast, lunch, dinner, snack) = EnumSort(’meals’, [’breakfast’, ’lunch’, ’dinner’, ’snack’]) \n\nfoods_sort, (fish, hot_cakes, macaroni, omelet, poached_eggs) = EnumSort(’foods’, [’fish’, ’hot_cakes’, ’macaroni’, ’omelet’, ’poached_eggs’]) \n\npeople = [Vladimir, Wendy] \n\nmeals = [breakfast, lunch, dinner, snack] \n\nfoods = [fish, hot_cakes, macaroni, omelet, poached_eggs] \n\neats = Function(’eats’, people_sort, meals_sort, foods_sort) \n\npre_conditions = [] \n\n# CONSTRAINT: At no meal does Vladimir eat the same kind of food as Wendy. \n\nm = Const(’m’, meals_sort) \n\npre_conditions.append(ForAll([m], eats(Vladimir, m) != eats(Wendy, m))) \n\nNewConstraint: \n\nNeither of them eats the same kind of food more than once during the day. \n\nNewConstraintCode: \n\nm = Const(’m’, meals_sort) \n\np = Const(’p’, people_sort) \n\nf = Const(’f’, foods_sort) \n\npre_conditions.append(ForAll([p, f], Sum([eats(p, m) == f for m in meals]) <= 1)) \n\n------ \n\nExistingProgram: \n\n# In a repair facility there are exactly six technicians: Stacy, Urma, Wim, Xena, Yolanda, and Zane. Each technician repairs machines of at least one of the following three types—radios, televisions, and VCRs—and no other types. \n\nfrom z3 import * \n\ntechnicians_sort, (Stacy, Urma, Wim, Xena, Yolanda, Zane) = EnumSort(’technicians’, [’Stacy’, ’Urma’, ’Wim’, ’Xena’, ’Yolanda’, ’Zane’]) \n\nmachines_sort, (radios, televisions, VCRs) = EnumSort(’machines’, [’radios’, ’televisions’, ’VCRs’]) \n\ntechnicians = [Stacy, Urma, Wim, Xena, Yolanda, Zane] \n\nmachines = [radios, televisions, VCRs] \n\nrepairs = Function(’repairs’, technicians_sort, machines_sort, BoolSort()) \n\npre_conditions = [] \n\nt = Const(’t’, technicians_sort) \n\npre_conditions.append(ForAll([t], Sum([repairs(t, m) for m in machines]) >= 1)) \n\nNewConstraint: \n\nXena and exactly three other technicians repair radios. \n\nNewConstraintCode: \n\nt = Const(’t’, technicians_sort) \n\npre_conditions.append(And(repairs(Xena, radios), Sum([And(t != Xena, repairs(t, radios)) for t in technicians]) == 3)) \n\n------', 'options': 'Given a problem with multiple answer options and an existing z3 program that models the problem, please provide the z3 code that checks each option and prints the correct answer. For each option, first create the check_property for the option by substituting the option values appropriately in the question statement, as well as a full comment describing what the check_property is stating. Then use only the following custom functions (is_unsat(), is_sat() and is_valid()) to check if the check_property is unsatisfiable, satisfiable or valid (depending on the question). Please structure the code with comments exactly as shown in the few shot examples below. Please provide only the options code and its comments in the output (not the full program), and no other surrounding markdown formatting or explanatory text. Please create independently executable code for each option (even if the option is not satisfiable) and do not share code between different options. \n\ndef is_unsat(option_constraints): \n\nsolver = Solver() \n\nsolver.add(pre_conditions) \n\nsolver.add(option_constraints) \n\nreturn solver.check() == unsat \n\ndef is_sat(option_constraints): \n\nsolver = Solver() \n\nsolver.add(pre_conditions) \n\nreturn solver.check() == sat \n\ndef is_valid(option_constraints): \n\nreturn is_sat(option_constraints) and is_unsat(Not(option_constraints)) \n\n------ \n\n>>> Problem: \n\nOn Tuesday Vladimir and Wendy each eat exactly four separate meals: breakfast, lunch, dinner, and a snack. \n\n>>> ExistingProgram: \n\nfrom z3 import * \n\npeople_sort, (Vladimir, Wendy) = EnumSort(’people’, [’Vladimir’, ’Wendy’]) \n\nmeals_sort, (breakfast, lunch, dinner, snack) = EnumSort(’meals’, [’breakfast’, ’lunch’, ’dinner’, ’snack’]) \n\nfoods_sort, (fish, hot_cakes, macaroni, omelet, poached_eggs) = EnumSort(’foods’, [’fish’, ’hot_cakes’, ’macaroni’, ’omelet’, ’poached_eggs’]) \n\npeople = [Vladimir, Wendy] \n\nmeals = [breakfast, lunch, dinner, snack] \n\nfoods = [fish, hot_cakes, macaroni, omelet, poached_eggs] \n\neats = Function(’eats’, people_sort, meals_sort, foods_sort) \n\npre_conditions = [] \n\n# CONSTRAINT: At no meal does Vladimir eat the same kind of food as Wendy. \n\nm = Const(’m’, meals_sort) \n\npre_conditions.append(ForAll([m], eats(Vladimir, m) != eats(Wendy, m))) \n\n# CONSTRAINT: Neither of them eats the same kind of food more than once during the day. \n\nm = Const(’m’, meals_sort) \n\np = Const(’p’, people_sort) \n\nf = Const(’f’, foods_sort) \n\npre_conditions.append(ForAll([p, f], Sum([eats(p, m) == f for m in meals]) <= 1)) \n\n>>> Question: \n\nVladimir cannot eat which one of the following foods? \n\n>>> Options: \n\n(A) fish \n\n(B) hot cakes \n\n(C) macaroni \n\n(D) omelet \n\n(E) poached eggs \n\n>>> OptionsCode: \n\n# CHECK TYPE: question says "cannot" so will check for validity using is_valid() to ensure that the negated statement is true in all possible models. \n\n# OPTION A: \n\n# CHECK PROPERTY: Vladimir cannot eat which one of the following foods? ANSWER: fish. \n\nm = Const(’m’, meals_sort) \n\ncheck_property = ForAll([m], eats(Vladimir, m) != fish) \n\nif is_valid(check_property): print(’(A)’) \n\n# OPTION B: \n\n# CHECK PROPERTY: Vladimir cannot eat which one of the following foods? ANSWER: hot cakes. \n\nm = Const(’m’, meals_sort) \n\ncheck_property = ForAll([m], eats(Vladimir, m) != hot_cakes) \n\nif is_valid(check_property): print(’(B)’) \n\n# OPTION C: \n\n# CHECK PROPERTY: Vladimir cannot eat which one of the following foods? ANSWER: macaroni. \n\nm = Const(’m’, meals_sort) \n\ncheck_property = ForAll([m], eats(Vladimir, m) != macaroni) \n\nif is_valid(check_property): print(’(C)’) \n\n# OPTION D: \n\n# CHECK PROPERTY: Vladimir cannot eat which one of the following foods? ANSWER: omelet. \n\nm = Const(’m’, meals_sort) \n\ncheck_property = ForAll([m], eats(Vladimir, m) != omelet) \n\nif is_valid(check_property): print(’(D)’) \n\n# OPTION E: \n\n# CHECK PROPERTY: Vladimir cannot eat which one of the following foods? ANSWER: poached eggs. \n\nm = Const(’m’, meals_sort) \n\ncheck_property = ForAll([m], eats(Vladimir, m) != poached_eggs) \n\nif is_valid(check_property): print(’(E)’) \n\n------ \n\n>>> Problem: \n\nIn a repair facility there are exactly six technicians: Stacy, Urma, Wim, Xena, Yolanda, and Zane. Each technician repairs equipment of at least one of the following three types—radios, televisions, and VCRs—and no other types. \n\n>>> ExistingProgram: \n\nfrom z3 import * \n\ntechnicians_sort, (Stacy, Urma, Wim, Xena, Yolanda, Zane) = EnumSort(’technicians’, [’Stacy’, ’Urma’, ’Wim’, ’Xena’, ’Yolanda’, ’Zane’]) \n\nequipment_sort, (radios, televisions, VCRs) = EnumSort(’equipment’, [’radios’, ’televisions’, ’VCRs’]) \n\ntechnicians = [Stacy, Urma, Wim, Xena, Yolanda, Zane] \n\nequipment = [radios, televisions, VCRs] \n\nrepairs = Function(’repairs’, technicians_sort, equipment_sort, BoolSort()) \n\npre_conditions = [] \n\nt = Const(’t’, technicians_sort) \n\npre_conditions.append(ForAll([t], Sum([repairs(t, e) for e in equipment]) >= 1)) \n\n# CONSTRAINT: Xena and exactly three other technicians repair radios. \n\nt = Const(’t’, technicians_sort) \n\npre_conditions.append(And(repairs(Xena, radios), Sum([And(t != Xena, repairs(t, radios)) for t in technicians]) == 3)) \n\n>>> Question: \n\nWhich one of the following can be a complete and accurate list of the technicians that repair televisions? \n\n>>> Options: \n\n(A) Stacy, Wim, Zane \n\n(B) Urma, Wim, Xena, Yolanda \n\n(C) Xena, Yolanda \n\n(D) Stacy, Urma, Wim, Xena, Yolanda, Zane \n\n(E) Urma \n\n>>> OptionsCode: \n\n# CHECK TYPE: question says "can be" so will check for satisfiable using is_sat() \n\n# OPTION A: \n\n# CHECK PROPERTY: Which one of the following can be a complete and accurate list of the technicians that repair televisions? ANSWER: Stacy, Wim and Zane. \n\ne = Const(’e’, equipment_sort) \n\ncheck_property = And(repairs(Stacy, televisions), repairs(Wim, televisions), repairs(Wim, televisions), Not(repairs(Urma, televisions)), Not(repairs(Xena, televisions)), Not(repairs(Yolanda, televisions))) \n\nif is_sat(check_property): print(’(A)’) \n\n# OPTION B: \n\n# CHECK PROPERTY: Which one of the following can be a complete and accurate list of the technicians that repair televisions? ANSWER: Urma, Wim, Xena and Yolanda. \n\ne = Const(’e’, equipment_sort) \n\ncheck_property = And(repairs(Urma, televisions), repairs(Wim, televisions), repairs(Xena, televisions), repairs(Yolanda, televisions), Not(repairs(Stacy, televisions)), Not(repairs(Zane, televisions))) \n\nif is_sat(check_property): print(’(B)’) \n\n# OPTION C: \n\n# CHECK PROPERTY: Which one of the following can be a complete and accurate list of the technicians that repair televisions? ANSWER: Xena and Yolanda. \n\ne = Const(’e’, equipment_sort) \n\ncheck_property = And(repairs(Xena, televisions), repairs(Yolanda, televisions), Not(repairs(Stacy, televisions)), Not(repairs(Urma, televisions)), Not(repairs(Wim, televisions)), Not(repairs(Zane, televisions))) \n\nif is_sat(check_property): print(’(C)’) \n\n# OPTION D: \n\n# CHECK PROPERTY: Which one of the following can be a complete and accurate list of the technicians that repair televisions? ANSWER: Stacy, Urma, Wim, Xena, Yolanda and Zane. \n\ne = Const(’e’, equipment_sort) \n\ncheck_property = And(repairs(Stacy, televisions), repairs(Urma, televisions), repairs(Wim, televisions), repairs(Xena, televisions), repairs(Yolanda, televisions), repairs(Zane, televisions)) \n\nif is_sat(check_property): print(’(D)’) \n\n# OPTION E: \n\n# CHECK PROPERTY: Which one of the following can be a complete and accurate list of the technicians that repair televisions? ANSWER: Urma. \n\ne = Const(’e’, equipment_sort) \n\ncheck_property = And(repairs(Urma, televisions), Not(repairs(Stacy, televisions)), Not(repairs(Wim, televisions)), Not(repairs(Xena, televisions)), Not(repairs(Yolanda, televisions)), Not(repairs(Zane, televisions))) \n\nif is_sat(check_property): print(’(E)’) \n\n------', 'error_repair': 'We are given a z3 program and an error message obtained from running it. First, please provide an analysis that investigates what may be the problem in the program that may be causing the error. Then, based on this analysis, please provide the corrected program where the issue is fixed - please make sure to retain any comments from the original code in the repaired code (especially the "CONSTRAINT", "QUESTION" or "OPTION" comments which demarcate special code segments - please do not remove, change or add any new such comments). If there is a general issue in the formulation, then please consider an alternative reformulation so that the program can execute without errors. A couple of sample cases are shown below for illustration. Please produce output in exactly the format shown in these samples, with the ">>> CorrectedProgram:" label clearly demarcating the corrected code, and do not use any other markdown formatting. \n\n------ \n\n>>> ExistingProgram: \n\nfrom z3 import * \n\npeople_sort, (Vladimir, Wendy) = EnumSort(’people’, [’Vladimir’, ’Wendy’]) \n\nmeals_sort, (breakfast, lunch, dinner, snack) = EnumSort(’meals’, [’breakfast’, ’lunch’, ’dinner’, ’snack’]) \n\nfoods_sort, (fish, hot_cakes, macaroni, omelet, poached_eggs) = EnumSort(’foods’, [’fish’, ’hot_cakes’, ’macaroni’, ’omelet’, ’poached_eggs’]) \n\npeople = [Vladimir, Wendy] \n\nmeals = [breakfast, lunch, dinner, snack] \n\nfoods = [fish, hot_cakes, macaroni, omelet, poached_eggs] \n\neats = Function(’eats’, people_sort, meals_sort, foods_sort) \n\npre_conditions = [] \n\n# CONSTRAINT: At no meal does Vladimir eat the same kind of food as Wendy. \n\npre_conditions.append(ForAll([m], eats(Vladimir, m) != eats(Wendy, m))) \n\n>>> ErrorMessage: \n\n"NameError: name ’m’ is not defined" \n\n>>> ProblemDiscussion: \n\nThis program defines three enumerations for people, meals, and foods. It then specifies that Vladimir and Wendy are people, and lists the available meals and foods. It also creates a function eats which represents the food each person eats at each meal. Finally, it tries to add a constraint to ensure that Vladimir and Wendy do not eat the same kind of food at any meal. However, as the error message indicates, the constraint code uses a variable ’m’ that has not been previously declared. Hence the correct fix to this issue would be to first explicitly declare the variable ’m’ as a new const of meal_sort. \n\n>>> CorrectedProgram: \n\nfrom z3 import * \n\npeople_sort, (Vladimir, Wendy) = EnumSort(’people’, [’Vladimir’, ’Wendy’]) \n\nmeals_sort, (breakfast, lunch, dinner, snack) = EnumSort(’meals’, [’breakfast’, ’lunch’, ’dinner’, ’snack’]) \n\nfoods_sort, (fish, hot_cakes, macaroni, omelet, poached_eggs) = EnumSort(’foods’, [’fish’, ’hot_cakes’, ’macaroni’, ’omelet’, ’poached_eggs’]) \n\npeople = [Vladimir, Wendy] \n\nmeals = [breakfast, lunch, dinner, snack] \n\nfoods = [fish, hot_cakes, macaroni, omelet, poached_eggs] \n\neats = Function(’eats’, people_sort, meals_sort, foods_sort) \n\npre_conditions = [] \n\n# CONSTRAINT: At no meal does Vladimir eat the same kind of food as Wendy. \n\nm = Const(’m’, meals_sort) \n\npre_conditions.append(ForAll([m], eats(Vladimir, m) != eats(Wendy, m))) \n\n------', 'instantiations': 'Given a problem scenario, some Z3 initialization code that defines the data structures, and a list of constraints, please provide positive and negative examples for each constraint. Each positive example should have a description and an expression of concrete assignments that satisfy the constraint, while each negative example should have a description and an expression of concrete assignments that contradict the constraint. If a constraint or its examples cannot be expressed by the given data structures or definitions, then please state "NONE" for the example description and "pass" for the assignments code. Please provide the completion to the prompt in exactly the same format as the example given below. \n\n------ \n\n>>> Scenario: \n\nNone \n\n>>> InitializationCode: \n\nfrom z3 import * \n\ncreature_sort = DeclareSort(’creature’) \n\nStella = Const(’Stella’, creature_sort) \n\nJay = Const(’Jay’, creature_sort) \n\nis_tumpus = Function(’is_tumpus’, creature_sort, BoolSort()) \n\nis_rompus = Function(’is_rompus’, creature_sort, BoolSort()) \n\nis_numpus = Function(’is_numpus’, creature_sort, BoolSort()) \n\nis_yumpus = Function(’is_yumpus’, creature_sort, BoolSort()) \n\nis_zumpus = Function(’is_zumpus’, creature_sort, BoolSort()) \n\nis_impus = Function(’is_impus’, creature_sort, BoolSort()) \n\nis_dumpus = Function(’is_dumpus’, creature_sort, BoolSort()) \n\nis_vumpus = Function(’is_vumpus’, creature_sort, BoolSort()) \n\nis_jompus = Function(’is_jompus’, creature_sort, BoolSort()) \n\nis_wumpus = Function(’is_wumpus’, creature_sort, BoolSort()) \n\nis_angry = Function(’is_angry’, creature_sort, BoolSort()) \n\nis_bright = Function(’is_bright’, creature_sort, BoolSort()) \n\nis_luminous = Function(’is_luminous’, creature_sort, BoolSort()) \n\nis_transparent = Function(’is_transparent’, creature_sort, BoolSort()) \n\nis_bitter = Function(’is_bitter’, creature_sort, BoolSort()) \n\nis_red = Function(’is_red’, creature_sort, BoolSort()) \n\nis_happy = Function(’is_happy’, creature_sort, BoolSort()) \n\nis_large = Function(’is_large’, creature_sort, BoolSort()) \n\npre_conditions = [] \n\n>>> Constraints: \n\nEach dumpus is a vumpus. \n\n### \n\nVumpuses are bright. \n\n### \n\nEvery vumpus is a zumpus. \n\n### \n\nZumpuses are not luminous. \n\n>>> ConstraintExamples: \n\nConstraint: \n\nEach dumpus is a vumpus. \n\nPositiveExampleDescription: \n\nStella is a dumpus and is also a vumpus. \n\nPositiveExampleCode: \n\nAnd(is_dumpus(Stella) == True, is_vumpus(Stella) == True) \n\nNegativeExampleDescription: \n\nStella is a dumpus but is not a vumpus. \n\nNegativeExampleCode: \n\nAnd(is_dumpus(Stella) == True, is_vumpus(Stella) == False) \n\nConstraint: \n\nVumpuses are bright. \n\nPositiveExampleDescription: \n\nJay is a vumpus and is bright. \n\nPositiveExampleCode: \n\nAnd(is_vumpus(Jay) == True, is_bright(Jay) == True) \n\nNegativeExampleDescription: \n\nJay is a vumpus and is not bright. \n\nNegativeExampleCode: \n\nAnd(is_vumpus(Jay) == True, is_bright(Jay) == False) \n\nConstraint: \n\nEvery vumpus is a zumpus. \n\nPositiveExampleDescription: \n\nJay is a vumpus and a zumpus. \n\nPositiveExampleCode: \n\nAnd(is_vumpus(Jay) == True, is_zumpus(Jay) == True) \n\nNegativeExampleDescription: \n\nJay is a vumpus but not a zumpus. \n\nNegativeExampleCode: \n\nAnd(is_vumpus(Jay) == True, is_zumpus(Jay) == False) \n\nConstraint: \n\nZumpuses are not luminous. \n\nPositiveExampleDescription: \n\nStella is a zumpus and is not luminous. \n\nPositiveExampleCode: \n\nAnd(is_zumpus(Stella) == True, is_luminous(Stella) == False) \n\nNegativeExampleDescription: \n\nStella is a zumpus and is luminous. \n\nNegativeExampleCode: \n\nAnd(is_zumpus(Stella) == True, is_luminous(Stella) == True) \n\n------ \n\n>>> Scenario: \n\nOn Tuesday Vladimir and Wendy each eat exactly two separate meals: breakfast and dinner. \n\n>>> InitializationCode: \n\nfrom z3 import * \n\npeople_sort, (Vladimir, Wendy) = EnumSort(’people’, [’Vladimir’, ’Wendy’]) \n\nmeals_sort, (breakfast, dinner) = EnumSort(’meals’, [’breakfast’, ’dinner’]) \n\nfoods_sort, (fish, hot_cakes, macaroni, omelet, poached_eggs) = EnumSort(’foods’, [’fish’, ’hot_cakes’, ’macaroni’, ’omelet’, ’poached_eggs’]) \n\npeople = [Vladimir, Wendy] \n\nmeals = [breakfast, dinner] \n\nfoods = [fish, hot_cakes, macaroni, omelet, poached_eggs] \n\neats = Function(’eats’, people_sort, meals_sort, foods_sort) \n\npre_conditions = [] \n\n>>> Constraints: \n\nAt no meal does Vladimir eat the same kind of food as Wendy. \n\n### \n\nNeither of them eats the same kind of food more than once during the day. \n\n### \n\nFor breakfast, each eats hot cakes. \n\n>>> ConstraintExamples: \n\nConstraint: \n\nAt no meal does Vladimir eat the same kind of food as Wendy. \n\nPositiveExampleDescription: \n\nVladimir and Wendy eat different foods at each meal: Vladimir has fish for breakfast while Wendy has hot cakes, and for dinner, Vladimir eats macaroni while Wendy has omelet. \n\nPositiveExampleCode: \n\nAnd(eats(Vladimir, breakfast) == fish, eats(Wendy, breakfast) == hot_cakes, \n\neats(Vladimir, dinner) == macaroni, eats(Wendy, dinner) == omelet) \n\nNegativeExampleDescription: \n\nAt dinner, both Vladimir and Wendy eat the same food, macaroni. \n\nNegativeExampleCode: \n\nAnd(eats(Vladimir, dinner) == macaroni, eats(Wendy, dinner) == macaroni) \n\nConstraint: \n\nNeither of them eats the same kind of food more than once during the day. \n\nPositiveExampleDescription: \n\nVladimir eats different foods for breakfast and dinner: fish for breakfast and hot cakes for dinner. Wendy also eats different foods for both meals: hot cakes for breakfast and omelet for dinner. \n\nPositiveExampleCode: \n\nAnd(eats(Vladimir, breakfast) == fish, eats(Vladimir, dinner) == hot_cakes, \n\neats(Wendy, breakfast) == hot_cakes, eats(Wendy, dinner) == omelet) \n\nNegativeExampleDescription: \n\nVladimir eats fish for both breakfast and dinner. \n\nNegativeExampleCode: \n\nAnd(eats(Vladimir, breakfast) == fish, eats(Vladimir, dinner) == fish) \n\nConstraint: \n\nFor breakfast, each eats hot cakes. \n\nPositiveExampleDescription: \n\nVladimir and Wendy both eat hot cakes for breakfast. \n\nPositiveExampleCode: \n\nAnd(eats(Vladimir, breakfast) == hot_cakes, eats(Wendy, breakfast) == hot_cakes) \n\nNegativeExampleDescription: \n\nVladimir eats macaroni for breakfast. \n\nNegativeExampleCode: \n\neats(Vladimir, breakfast) == macaroni \n\n------ \n\n>>> Scenario: \n\nIn a repair facility there are exactly six technicians: Stacy, Urma, Wim, Xena, Yolanda, and Zane. Each technician repairs machines of at least one of the following three types—radios, televisions, and VCRs—and no other types. \n\n>>> InitializationCode: \n\nfrom z3 import * \n\ntechnicians_sort, (Stacy, Urma, Wim, Xena, Yolanda, Zane) = EnumSort(’technicians’, [’Stacy’, ’Urma’, ’Wim’, ’Xena’, ’Yolanda’, ’Zane’]) \n\nmachines_sort, (radios, televisions, VCRs) = EnumSort(’machines’, [’radios’, ’televisions’, ’VCRs’]) \n\ntechnicians = [Stacy, Urma, Wim, Xena, Yolanda, Zane] \n\nmachines = [radios, televisions, VCRs] \n\nrepairs = Function(’repairs’, technicians_sort, machines_sort, BoolSort()) \n\npre_conditions = [] \n\nt = Const(’t’, technicians_sort) \n\npre_conditions.append(ForAll([t], Sum([repairs(t, m) for m in machines]) >= 1)) \n\n>>> Constraints: \n\nXena and exactly three other technicians repair radios. \n\n### \n\nStacy needs help repairing VCRs. \n\n### \n\nUrma and Zane repair the same type of machine. \n\n>>> ConstraintExamples: \n\nConstraint: \n\nXena and exactly three other technicians repair radios. \n\nPositiveExampleDescription: \n\nOnly Xena, Wim, Yolanda, and Zane repair radios and no one else. \n\nPositiveExampleCode: \n\nAnd(repairs(Stacy, radios) == False, repairs(Urma, radios) == False, repairs(Wim, radios) == True, repairs(Xena, radios) == True, repairs(Yolanda, radios) == True, repairs(Zane, radios) == True) \n\nNegativeExampleDescription: \n\nOnly Xena and Yolanda repair radios and no one else. \n\nNegativeExampleCode: \n\nAnd(repairs(Stacy, radios) == False, repairs(Urma, radios) == False, repairs(Wim, radios) == False, repairs(Xena, radios) == True, repairs(Yolanda, radios) == True, repairs(Zane, radios) == False) \n\nConstraint: \n\nStacy needs help repairing VCRs. \n\nPositiveExampleDescription: \n\nNONE \n\nPositiveExampleCode: \n\npass \n\nNegativeExampleDescription: \n\nNONE \n\nNegativeExampleCode: \n\npass \n\nConstraint: \n\nUrma and Zane repair the same type of machine. \n\nPositiveExampleDescription: \n\nUrma and Zane both repair VCRs. \n\nPositiveExampleCode: \n\nAnd(repairs(Urma, VCRs) == True, repairs(Zane, VCRs) == True) \n\nNegativeExampleDescription: \n\nUrma repairs televisions, while Zane repairs radios. \n\nNegativeExampleCode: \n\nAnd(repairs(Urma, televisions) == True, repairs(Zane, radios) == True) \n\n------', 'semantic_repair': 'We are given a scenario description, some initial z3 code that sets up basic definitions, a constraint in natural language, and a code snippet that implements that constraint. We are also given some code that should implement a positive example to the constraint, which should be satisfiable under that constraint, but it is not. First, please provide an analysis that investigates what may be the problem in either the initial code, the constraint code or the example. Then, based on this analysis, please repair the relevant code segments (initial code, constraint code, or example code) so that the positive example becomes satisfiable (state ’NONE’ if no repair is required to a code segment). If multiple segments are incorrect due to a general formulation problem, then please reformulate the whole solution approach in the initial code and produce appropriate code for all segments. A couple of sample cases are shown below for illustration. Please produce output in exactly the format shown in these samples, and do not use any other markdown formatting. \n\n------ \n\nScenario: \n\nOn Tuesday Vladimir and Wendy each eat exactly four separate meals: breakfast, lunch, dinner, and a snack. \n\nInitialCode: \n\nfrom z3 import * \n\npeople_sort, (Vladimir, Wendy) = EnumSort(’people’, [’Vladimir’, ’Wendy’]) \n\nmeals_sort, (breakfast, lunch, dinner, snack) = EnumSort(’meals’, [’breakfast’, ’lunch’, ’dinner’, ’snack’]) \n\nfoods_sort, (fish, hot_cakes, macaroni, omelet, poached_eggs) = EnumSort(’foods’, [’fish’, ’hot_cakes’, ’macaroni’, ’omelet’, ’poached_eggs’]) \n\npeople = [Vladimir, Wendy] \n\nmeals = [breakfast, lunch, dinner, snack] \n\nfoods = [fish, hot_cakes, macaroni, omelet, poached_eggs] \n\neats = Function(’eats’, people_sort, meals_sort, foods_sort) \n\npre_conditions = [] \n\nConstraintDescription: \n\nAt some meal Vladimir eats the same kind of food as Wendy. \n\nConstraintCode: \n\nm = Const(’m’, meals_sort) \n\npre_conditions.append(ForAll([m], eats(Vladimir, m) != eats(Wendy, m))) \n\nPositiveExampleCode: \n\nAnd(eats(Vladimir, breakfast) == fish, eats(Wendy, breakfast) == fish) \n\nProblemDiscussion: \n\nThe scenario describes foods that Vladimir and Wendy eat at various meals during the day. The initial code defines the main data structures and the eats function which indicates the food each person eats on every meal. The constraint requires that there is at least one meal where they both eat the same food. The constraint code asserts that for all meals, the food that Vladimir eats is different from what Wendy eats. But this contradicts the intended constraint. The positive example code states that at breakfast, both Vladimir and Wendy eat fish, and this is consistent with the requirements of the constraint. Hence there is no issue in the initial code and the example code, but the constraint code wrongly implements the constraint. It should be repaired to assert that for some meal, both Vladimir and Wendy eat the same food. \n\nRepairedInitialCode: \n\nNONE \n\nRepairedConstraintCode: \n\nm = Const(’m’, meals_sort) \n\npre_conditions.append(Exists([m], eats(Vladimir, m) == eats(Wendy, m))) \n\nRepairedPositiveExampleCode: \n\nNONE \n\n------ \n\nScenario: \n\nIn a repair facility there are exactly six technicians: Stacy, Urma, Wim, Xena, Yolanda, and Zane. Each technician repairs machines of at least one of the following three types—radios, televisions, and VCRs—and no other types. \n\nInitialCode: \n\nfrom z3 import * \n\ntechnicians_sort, (Stacy, Urma, Wim, Xena, Yolanda, Zane) = EnumSort(’technicians’, [’Stacy’, ’Urma’, ’Wim’, ’Xena’, ’Yolanda’, ’Zane’]) \n\nmachines_sort, (radios, televisions, VCRs) = EnumSort(’machines’, [’radios’, ’televisions’, ’VCRs’]) \n\ntechnicians = [Stacy, Urma, Wim, Xena, Yolanda, Zane] \n\nmachines = [radios, televisions, VCRs] \n\nrepairs = Function(’repairs’, technicians_sort, machines_sort, BoolSort()) \n\npre_conditions = [] \n\nt = Const(’t’, technicians_sort) \n\npre_conditions.append(ForAll([t], Sum([repairs(t, m) for m in machines]) <= 1)) \n\nConstraintDescription: \n\nUrma repairs radios and VCRs \n\nConstraintCode: \n\npre_conditions.append(And(repairs(Urma, radios), repairs(Urma, VCRs))) \n\nPositiveExampleCode: \n\nAnd(repairs(Urma, radios) == True, repairs(Urma, VCRs) == True) \n\nProblemDiscussion: \n\nThe scenario describes types of machines that technicians repair at a repair facility, where each technician repairs at least one type of machine. The initial code defines the main data structures and the repairs function which indicates the type of machine repaired by each technician. It also adds the general condition that each technician can repair at most one type of machine, which is an incorret interpretation of the scenario statement that each technician must repair AT LEAST one type of machine. The constraint requires that Urma repairs both VCRs and radios. The constraint code correctly asserts this requirement, and the positive example code also states this correctly. Hence there is no issue in the constraint code and the example code, but the initial code wrongly prevents any technician from repairing two kinds of machines. It should be repaired to assert that each technician must repair at least one kind of machine. \n\nRepairedInitialCode: \n\nfrom z3 import * \n\ntechnicians_sort, (Stacy, Urma, Wim, Xena, Yolanda, Zane) = EnumSort(’technicians’, [’Stacy’, ’Urma’, ’Wim’, ’Xena’, ’Yolanda’, ’Zane’]) \n\nmachines_sort, (radios, televisions, VCRs) = EnumSort(’machines’, [’radios’, ’televisions’, ’VCRs’]) \n\ntechnicians = [Stacy, Urma, Wim, Xena, Yolanda, Zane] \n\nmachines = [radios, televisions, VCRs] \n\nrepairs = Function(’repairs’, technicians_sort, machines_sort, BoolSort()) \n\npre_conditions = [] \n\nt = Const(’t’, technicians_sort) \n\npre_conditions.append(ForAll([t], Sum([repairs(t, m) for m in machines]) >= 1)) \n\nRepairedConstraintCode: \n\nNONE \n\nRepairedPositiveExampleCode: \n\nNONE \n\n------'}

RULES = """You formalize a bounded evidence-based task, not general medical advice.
Only the supplied, actually acquired public observations are evidence. Claims
are questions, not premises. Keep independent scenarios and historical/current
facts separate. A recommendation is not an executed action. Missing current
values stay unconstrained; do not fill them using history or outside knowledge.
Original source rules apply only when their stated conditions apply. A rule
whose applicability is unknown does not establish its consequent by itself.
Preserve source wording and units when identifying constraints. Prior proposed
labels are not ground truth. Do not read files, call APIs, or import anything
except z3 in generated code. Use Python Z3 expressions, not Python and/or/not.
"""

PROGRAM_FORMAT = """Return one JSON object with these fields:
{
 "initial_context": "natural-language domain setup",
 "init": "from z3 import *\\n...declarations...\\npre_conditions = []",
 "constraints": [{"id":"K1", "text":"one original premise/rule",
                  "code":"pre_conditions.append(Z3_expression)"}],
 "queries": [{"item_id":"original item id", "text":"original claim",
              "code":"check_property = Z3_expression"}]
}
init contains shared symbols, domain setup, and any required domain constraints.
Each constraint code segment must append its own constraint to pre_conditions;
all declarations it uses must already exist in init. Each query must assign a
Boolean Z3 expression to check_property. Do not return labels or a solver loop.
Include every requested claim exactly once. Encode numerical relations as real
or integer arithmetic, not opaque predicate names whenever they are arithmetic.
Statements of missing information impose no value constraints; still define
the corresponding symbols so both possible truth values can be investigated.
Do not force unknown claims true or false just to obtain an answer.
"""

INSTANTIATIONS = """From the natural-language setup and descriptions below, create
one concrete positive and one concrete negative example for EVERY entry.
You are not shown the implementation of those entries. Derive examples from
their natural-language meaning, using only the supplied initialization symbols.
A positive example must concretely satisfy the entry, a negative one concretely
violate it. Use assignments/literals, not a repetition of the abstract rule.
For a conditional, the negative example should satisfy its antecedent and
violate its consequent. Keep scenarios distinct. Both examples must themselves
be possible under the initial domain setup. If the symbols cannot express an
example, use null and describe why instead of inventing a new symbol.
Return JSON {"examples":[{"id":"entry id",
"positive":{"description":"...","expression":"Z3 Boolean expression"},
"negative":{"description":"...","expression":"Z3 Boolean expression"}}]}.
"""


def _json(text: str) -> dict:
    # JSON mode is requested. Do not silently repair malformed model JSON.
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("ssv_json_not_object")
    return obj


def _ask(ctx, prompt: str, data: Any, stage: str, temperature: float) -> str:
    reference = None
    if "syntax_repair" in stage:
        reference = "error_repair"
    elif "semantic_repair" in stage:
        reference = "semantic_repair"
    elif stage.endswith("_instantiations"):
        reference = "instantiations"
    elif stage.endswith("_decompose"):
        reference = "decomposition"
    elif stage.endswith("_initialization") or "_constraint_" in stage:
        reference = "incremental"
    elif stage.endswith("_queries") or stage.endswith("_direct"):
        reference = "options"
    if reference:
        prompt = ("Published SSV reference prompt and complete demonstrations:\n"
                  + PUBLISHED_PROMPTS[reference]
                  + "\n\nCURRENT TASK ADAPTATION: Apply that method to the task below. "
                    "The following JSON transport instructions replace the demonstrations' "
                    "plain-text output delimiters and printed option letters.\n" + prompt)
    return ctx.llm(
        [{"role": "system", "content": RULES},
         {"role": "user", "content": prompt + "\n" + json.dumps(data, ensure_ascii=False)}],
        stage="ssv_" + stage, temperature=temperature, json_mode=True,
    )


def _public_problem(problem: dict) -> dict:
    """Do not seed the formalizer with the baseline's unverified answer."""
    return {"task_input": problem.get("task_input"),
            "observations": problem.get("observations", [])}


def _claim_ids(problem: dict) -> list[str]:
    task = problem.get("task_input") or {}
    for key in ("claims", "items"):
        rows = task.get(key)
        if isinstance(rows, list) and rows and all(isinstance(x, dict) and "item_id" in x for x in rows):
            return [x["item_id"] for x in rows]
    action = problem.get("proposed_action") or {}
    arguments = action.get("arguments", action.get("args", action))
    return [x["item_id"] for x in arguments.get("items", []) if isinstance(x, dict) and "item_id" in x]


def _validate(program: dict, ids: list[str]) -> None:
    if not isinstance(program.get("init"), str):
        raise ValueError("ssv_missing_initialization")
    if not isinstance(program.get("initial_context"), str):
        raise ValueError("ssv_missing_initial_context")
    constraints, queries = program.get("constraints"), program.get("queries")
    if not isinstance(constraints, list) or not isinstance(queries, list) or not queries:
        raise ValueError("ssv_bad_program_segments")
    if any(not isinstance(c, dict) or not all(isinstance(c.get(k), str) for k in ("id", "text", "code")) for c in constraints):
        raise ValueError("ssv_bad_constraint_segment")
    if len({c["id"] for c in constraints}) != len(constraints):
        raise ValueError("ssv_duplicate_constraint_id")
    if any(not isinstance(q, dict) or not all(isinstance(q.get(k), str) for k in ("item_id", "text", "code")) for q in queries):
        raise ValueError("ssv_bad_query_segment")
    query_ids = [q["item_id"] for q in queries]
    if len(query_ids) != len(set(query_ids)) or (ids and set(query_ids) != set(ids)):
        raise ValueError("ssv_query_ids_differ_from_task")


def _program_code(program: dict) -> str:
    """Execute the formalized problem; missing facts are open-world, not false."""
    code = program["init"] + "\n"
    code += "\n".join(c["code"] for c in program["constraints"]) + "\n"
    code += """
ssv_solver = Solver()
ssv_solver.set(timeout=10000)
ssv_solver.add(pre_conditions)
ssv_base_status = str(ssv_solver.check())
result = {"base_status": ssv_base_status, "items": [], "queries": []}
"""
    for q in program["queries"]:
        code += "\n" + q["code"] + "\n"
        code += f"""
ssv_solver.push()
ssv_solver.add(check_property)
ssv_positive = str(ssv_solver.check())
ssv_solver.pop()
ssv_solver.push()
ssv_solver.add(Not(check_property))
ssv_negative = str(ssv_solver.check())
ssv_solver.pop()
ssv_label = None
if ssv_base_status == "sat":
    if ssv_positive == "sat" and ssv_negative == "unsat":
        ssv_label = "supported"
    elif ssv_positive == "unsat" and ssv_negative == "sat":
        ssv_label = "refuted"
    elif ssv_positive == "sat" and ssv_negative == "sat":
        ssv_label = "not_determined"
result["queries"].append({{"item_id": {q['item_id']!r}, "positive": ssv_positive, "negative": ssv_negative}})
if ssv_label is not None:
    result["items"].append({{"item_id": {q['item_id']!r}, "label": ssv_label}})
"""
    return code


def _execute(ctx, program: dict, stage: str) -> dict:
    execution = ctx.execute(_program_code(program), stage="ssv_" + stage)
    ctx.record(stage + "_execution", execution)
    return execution


def _complete_answer(execution: dict, program: dict) -> list[dict] | None:
    if not execution.get("ok"):
        return None
    result = execution.get("result")
    if not isinstance(result, dict) or result.get("base_status") != "sat":
        return None
    items = result.get("items", [])
    if len(items) != len(program["queries"]) or any(x.get("label") not in LABELS for x in items):
        return None
    return items


def _compositional(ctx, problem: dict, ids: list[str], temperature: float, stage: str) -> dict | None:
    raw = _ask(ctx, "Decompose this problem into an initial_context and independent natural-language "
               "constraints. Return JSON {initial_context: string, constraints: [{id, text}], "
               "queries: [{item_id,text}]}. Keep requested claims out of premises.", problem,
               stage + "_decompose", temperature)
    try:
        plan = _json(raw)
        constraints = plan["constraints"]
        queries = plan["queries"]
        initial_context = plan["initial_context"]
        if not isinstance(constraints, list) or not isinstance(queries, list):
            raise ValueError("ssv_bad_decomposition")
    except (ValueError, KeyError, TypeError) as error:
        ctx.record(stage + "_decomposition_error", {"error": str(error), "raw": raw})
        return None
    ctx.record(stage + "_decomposition", plan)
    init_raw = _ask(ctx, "Produce the shared Z3 declarations and initial domain setup for the plan. "
                   "Declare symbols required by all constraints and queries. Return JSON {init: string}. "
                   "Initialize pre_conditions=[]; include only domain setup, not individual constraints.",
                   plan, stage + "_initialization", temperature)
    try:
        init = _json(init_raw)["init"]
        if not isinstance(init, str):
            raise ValueError("ssv_bad_initialization")
    except (ValueError, KeyError, TypeError) as error:
        ctx.record(stage + "_initialization_error", {"error": str(error), "raw": init_raw})
        return None
    program = {"initial_context": initial_context, "init": init, "constraints": [], "queries": []}
    for index, c in enumerate(constraints):
        raw = _ask(ctx, "Extend the existing Z3 program with this single natural-language constraint. "
                   "Return JSON {code: string}; code appends to pre_conditions and does not replace "
                   "earlier constraints or declarations. Supply only the new segment.",
                   {"existing_program": program, "new_constraint": c},
                   stage + f"_constraint_{index}", temperature)
        try:
            program["constraints"].append({"id": c["id"], "text": c["text"], "code": _json(raw)["code"]})
        except (ValueError, KeyError, TypeError) as error:
            ctx.record(stage + "_constraint_error", {"index": index, "error": str(error), "raw": raw})
            return None
    raw = _ask(ctx, "Translate each requested claim into an independent Z3 query segment assigning "
               "check_property. Return JSON {queries:[{item_id,text,code}]}; do not assert queries as facts.",
               {"existing_program": program, "queries": queries}, stage + "_queries", temperature)
    try:
        program["queries"] = _json(raw)["queries"]
        _validate(program, ids)
    except (ValueError, KeyError, TypeError) as error:
        ctx.record(stage + "_queries_error", {"error": str(error), "raw": raw})
        return None
    return program


def _gen_program(ctx, problem: dict, ids: list[str], temperature: float, stage: str) -> tuple[dict | None, dict | None]:
    raw = _ask(ctx, "Formalize the complete public task as one segmented Z3 program.\n" + PROGRAM_FORMAT,
               problem, stage + "_direct", temperature)
    execution = None
    for attempt in range(MAX_REPAIRS + 1):
        try:
            program = _json(raw)
            _validate(program, ids)
        except (ValueError, KeyError, TypeError) as error:
            execution = {"ok": False, "error": str(error)}
            program = None
        else:
            ctx.record(stage + f"_program_{attempt}", program)
            execution = _execute(ctx, program, stage + f"_syntax_{attempt}")
            if execution.get("ok"):
                return program, execution
        if attempt < MAX_REPAIRS:
            raw = _ask(ctx, "Repair the JSON/program syntax or execution error. Retain all original "
                       "constraint and query identities and meanings. Return the complete corrected "
                       "program object.\n" + PROGRAM_FORMAT,
                       {"problem": problem, "previous_program": raw, "execution_error": execution},
                       stage + f"_syntax_repair_{attempt + 1}", temperature)
    program = _compositional(ctx, problem, ids, temperature, stage + "_compositional")
    if program is None:
        return None, execution
    ctx.record(stage + "_compositional_program", program)
    execution = _execute(ctx, program, stage + "_compositional")
    return (program, execution) if execution.get("ok") else (None, execution)


def _entries(program: dict) -> list[dict]:
    return [{"id": "constraint:" + c["id"], "text": c["text"], "code": c["code"], "kind": "constraint"}
            for c in program["constraints"]] + [
        {"id": "query:" + q["item_id"], "text": q["text"], "code": q["code"], "kind": "query"}
        for q in program["queries"]]


def _verification_code(program: dict, entry: dict, examples: dict) -> str:
    code = program["init"] + "\nssv_initial_conditions = list(pre_conditions)\n"
    code += "ssv_initial_count = len(pre_conditions)\n" + entry["code"] + "\n"
    if entry["kind"] == "constraint":
        code += "ssv_constraint = And(pre_conditions[ssv_initial_count:])\n"
    else:
        code += "ssv_constraint = check_property\n"
    code += "ssv_positive_example = (" + examples["positive"]["expression"] + ")\n"
    code += "ssv_negative_example = (" + examples["negative"]["expression"] + ")\n"
    code += """
def ssv_check(*expressions):
    s = Solver()
    s.set(timeout=10000)
    s.add(ssv_initial_conditions)
    s.add(*expressions)
    return str(s.check())
def ssv_nonvacuous(expression):
    if is_quantifier(expression):
        variables = [FreshConst(expression.var_sort(i)) for i in range(expression.num_vars())]
        return ssv_nonvacuous(substitute_vars(expression.body(), *reversed(variables)))
    if not is_app(expression):
        return True
    if expression.decl().kind() == Z3_OP_IMPLIES:
        antecedent, consequent = expression.children()
        if ssv_check(antecedent) != "sat" or ssv_check(Not(consequent)) != "sat":
            return False
    return all(ssv_nonvacuous(child) for child in expression.children())
result = {
    "positive_feasible": ssv_check(ssv_positive_example),
    "negative_feasible": ssv_check(ssv_negative_example),
    "positive_with_constraint": ssv_check(ssv_constraint, ssv_positive_example),
    "negative_with_constraint": ssv_check(ssv_constraint, ssv_negative_example),
    "constraint_possible": ssv_check(ssv_constraint),
    "constraint_falsifiable": ssv_check(Not(ssv_constraint)),
    "nested_implications_nonvacuous": ssv_nonvacuous(ssv_constraint),
}
result["passed"] = (
    result["positive_feasible"] == "sat"
    and result["negative_feasible"] == "sat"
    and result["positive_with_constraint"] == "sat"
    and result["negative_with_constraint"] == "unsat"
    and result["constraint_possible"] == "sat"
    and result["constraint_falsifiable"] == "sat"
    and result["nested_implications_nonvacuous"]
)
"""
    return code


def _verify(ctx, program: dict, temperature: float, stage: str,
            repaired_examples: list[dict] | None = None) -> tuple[bool, dict | None]:
    entries = _entries(program)
    # Deliberately hide constraint/query code from the independent example task.
    raw = _ask(ctx, INSTANTIATIONS,
               {"initial_context": program["initial_context"], "initialization_code": program["init"],
                "entries": [{"id": e["id"], "text": e["text"]} for e in entries]},
               stage + "_instantiations", temperature)
    try:
        value = _json(raw)
        example_rows = value["examples"]
        examples = {e["id"]: e for e in example_rows}
        if len(examples) != len(example_rows) or set(examples) != {e["id"] for e in entries}:
            raise ValueError("ssv_example_ids_differ")
    except (ValueError, KeyError, TypeError) as error:
        failure = {"kind": "instantiation_format", "error": str(error), "raw": raw}
        ctx.record(stage + "_verification_failure", failure)
        return False, failure
    ctx.record(stage + "_instantiations", value)
    if repaired_examples:
        for patch in repaired_examples:
            if not isinstance(patch, dict) or patch.get("id") not in examples:
                failure = {"kind": "invalid_repaired_example", "patch": patch}
                ctx.record(stage + "_verification_failure", failure)
                return False, failure
            examples[patch["id"]] = patch
        ctx.record(stage + "_repaired_examples", repaired_examples)
    for index, entry in enumerate(entries):
        example = examples[entry["id"]]
        if any(not isinstance(example.get(k), dict) or not isinstance(example[k].get("expression"), str)
               or example[k].get("description") in (None, "NONE") for k in ("positive", "negative")):
            failure = {"kind": "unexpressible_instantiation", "entry": entry, "examples": example}
            ctx.record(stage + "_verification_failure", failure)
            return False, failure
        execution = ctx.execute(_verification_code(program, entry, example),
                                stage="ssv_" + stage + f"_verify_{index}")
        ctx.record(stage + f"_verify_{index}", {"entry": entry, "examples": example, "execution": execution})
        if not execution.get("ok") or not isinstance(execution.get("result"), dict) or not execution["result"].get("passed"):
            failure = {"kind": "semantic_verification", "entry": entry, "examples": example, "execution": execution}
            ctx.record(stage + "_verification_failure", failure)
            return False, failure
    return True, None


def run(ctx) -> dict:
    problem = _public_problem(ctx.problem)
    ids = _claim_ids(ctx.problem)
    best_answer, best_stage = None, None
    attempts = []
    ctx.record("method_config", {"method": "ssv", "implementation": "paper_reconstruction",
               "temperatures": list(TEMPERATURES), "max_repairs": MAX_REPAIRS,
               "semantic_repair_only_without_answer": True,
               "source": "https://arxiv.org/html/2501.16961v3",
               "official_repository_status": "empty_on_2026-11-12"})
    for index, temperature in enumerate(TEMPERATURES):
        stage = f"temperature_{index}"
        program, execution = _gen_program(ctx, problem, ids, temperature, stage)
        if program is None:
            attempts.append({"temperature": temperature, "status": "no_executable_program"})
            continue
        repaired_examples = None
        for repair in range(MAX_REPAIRS + 1):
            answer = _complete_answer(execution, program)
            if best_answer is None and answer is not None:
                best_answer, best_stage = answer, stage
            verified, failure = _verify(ctx, program, temperature, stage + f"_semantic_{repair}",
                                        repaired_examples)
            attempts.append({"temperature": temperature, "repair": repair,
                             "has_answer": answer is not None, "verified": verified,
                             "failure_kind": failure.get("kind") if failure else None})
            if verified and answer is not None:
                result = {"method": "ssv", "status": "completed", "items": answer,
                          "diagnostics": {"verified": True, "attempts": attempts,
                                          "implementation": "paper_reconstruction"}}
                ctx.record("result", result)
                return result
            # Figure 5 line 14 and §2: semantic repair is attempted only when
            # there was no answer. Otherwise retain the first unverified answer
            # and explore the next temperature, rather than silently adding a
            # stronger repair policy not present in the paper.
            if answer is not None or repair >= MAX_REPAIRS:
                break
            raw = _ask(ctx, "A semantic instance or well-formedness check failed. Diagnose whether "
                       "the initialization, constraint/query translation, or example is wrong. "
                       "Repair the program consistently with the original evidence, keeping every "
                       "constraint/query identity. Return the complete program JSON. If example code "
                       "is wrong, also include example_repairs:[{id,positive:{description,expression},"
                       "negative:{description,expression}}] with the corrected pair. These examples "
                       "will still undergo solver checks.\n" + PROGRAM_FORMAT,
                       {"problem": problem, "program": program, "failure": failure, "execution": execution},
                       stage + f"_semantic_repair_{repair + 1}", temperature)
            try:
                program = _json(raw)
                _validate(program, ids)
                repaired_examples = program.get("example_repairs")
                if repaired_examples is not None and not isinstance(repaired_examples, list):
                    raise ValueError("ssv_bad_repaired_examples")
            except (ValueError, KeyError, TypeError) as error:
                ctx.record(stage + f"_semantic_repair_error_{repair + 1}", {"error": str(error), "raw": raw})
                break
            ctx.record(stage + f"_repaired_program_{repair + 1}", program)
            execution = _execute(ctx, program, stage + f"_repaired_{repair + 1}")
            if not execution.get("ok"):
                break
    if best_answer is not None:
        result = {"method": "ssv", "status": "completed_unverified", "items": best_answer,
                  "diagnostics": {"verified": False, "selected_first_answer_from": best_stage,
                                  "attempts": attempts, "implementation": "paper_reconstruction"}}
    else:
        # Figure 5: direct LLM fallback is part of the source method, not an
        # invented parser-error -> unknown conversion.
        raw = _ask(ctx, "No solver program has produced an answer. Work through the supplied "
                   "evidence and directly answer every claim. Return JSON {items:[{item_id,label}], "
                   "explanation:string}; labels are supported, refuted, not_determined. "
                   "Missing evidence can justify not_determined; program failure itself cannot.",
                   problem, "direct_fallback", 0.0)
        try:
            value = _json(raw)
            items = value["items"]
            if not isinstance(items, list) or any(not isinstance(i, dict) or i.get("label") not in LABELS for i in items):
                raise ValueError("ssv_invalid_fallback_items")
            if ids and {i.get("item_id") for i in items} != set(ids):
                raise ValueError("ssv_fallback_ids_differ")
            if len({i.get("item_id") for i in items}) != len(items):
                raise ValueError("ssv_fallback_duplicate_ids")
        except (ValueError, KeyError, TypeError) as error:
            result = {"method": "ssv", "status": "failed", "items": [],
                      "diagnostics": {"verified": False, "error": str(error), "attempts": attempts,
                                      "fallback": True, "implementation": "paper_reconstruction"}}
        else:
            result = {"method": "ssv", "status": "direct_llm_fallback", "items": items,
                      "diagnostics": {"verified": False, "attempts": attempts, "fallback": True,
                                      "implementation": "paper_reconstruction"}}
    ctx.record("result", result)
    return result
