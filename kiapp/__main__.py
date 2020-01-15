from kiapp import app
import argparse

if __name__ == '__main__':

    # initiate the parser
    parser = argparse.ArgumentParser()

    # add long and short argument
    parser.add_argument("--generate", "-g", help="Generate Model at specified path default kiapp directory", action="store_true")
    parser.add_argument("--evaluateFile", "-f", help="Evaluate specified file", action="store_true")

    # read arguments from the command line
    args = parser.parse_args()
    if(args.generate):
        app.run('generate')
    elif(args.evaluateFile):
        app.run('evaluateFile')
    else:
        app.run()
