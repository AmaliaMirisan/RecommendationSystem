# Script pentru transformarea unui șir de caractere într-o listă

def transform_string_to_list(choices):
    # Îndepărtează spațiile suplimentare și transformă șirul într-o listă
    choices_list = [choice.strip() for choice in choices.split(',')]
    return choices_list

# Exemplu de utilizare
choices_string = "ottawa, fairbanks, toronto, yellowknife, quebec_city, niagara_falls, seattle, whitehorse, squamish, okanagan_valley, canadian_rockies, whistler, vancouver_island, montreal, calgary, vancouver "
choices_list = transform_string_to_list(choices_string)
print(choices_list)
