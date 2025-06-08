import json
import parser
import my_ai

classifier = my_ai.create_model(
        data_path='code_dataset.csv',
        model_type='nn',
        epochs=250
    )

predictions = my_ai.check(classifier, ['print("HEELEHEAIU")'])

print("\nРезультаты классификации:")
for result in predictions:
    print(f"\nКод:\n{result['code']}\nПредсказание: {result['prediction']}")
