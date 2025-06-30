from typing import TypedDict #  it does not validate data


class Person(TypedDict):
    name : str
    age : int



new_person: Person = {'name' : "Hasan", "age" : 32}

print(new_person)