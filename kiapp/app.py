from kiapp.ki import Ki

ki = Ki()
running = True

def generate_model():
    global ki
    ki.generate_model()
    print('\n\n**********************\nDone Generating\n**********************\n\n')

def evaluate_file():
    global ki
    ki.evaluate_file()
    print('\n\n**********************\nDone Evaluating\n**********************\n\n')

def test_text():
    global ki
    test_running = True
    while(test_running):
        userInput = input("Please enter a review: ")
        result = ki.evaluate_text(userInput)
        print(result)
        stop = input("Another Review? y or n: ").lower()
        if(stop == 'n'):
            test_running = False


def stop_program():
    global running
    print("Stopping programm")
    running = False

def accuracy():
    global ki
    ki.accuracy()

allowed_args = {
    'generate': generate_model,
    'evaluateFile': evaluate_file,
}

commands = {
    "1": {
        'desc': 'Test: Input Text and see result',
        'func': test_text,
    },  
    '2': {
        'desc': 'Generate: Generate new model',
        'func': generate_model,
    },
    '3': {
        'desc': 'Evaluate: Evaluate file evaluation.txt in data folder',
        'func': evaluate_file,
    },
    '4': {
        'desc': 'Accuracy: Get accuracy score',
        'func': accuracy,
    },
    "exit": {
        'desc': 'Exit programm',
        'func': stop_program
    },
}

""" Main Programm routine """
def run(*argv):
    global running
    for arg in argv:  
        arg_function = allowed_args[arg]
        arg_function()

    while(running):
        for key, command  in commands.items():
            print("%s -  %s" % (key, command['desc']))
        
        not_valid_command = True
        userInput = 0
        #check for valid command
        while(not_valid_command):
            userInput = input('Enter command number:')
            if userInput in commands:
                not_valid_command = False

        command = commands[userInput]
        func = command['func']
        func()

