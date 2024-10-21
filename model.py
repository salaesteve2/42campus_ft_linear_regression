import argparse
import sys
import os
import time


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('mileage', help='mileage to check')
	parser.add_argument('file', help='file to train')
	parser.add_argument('-g', "--graph", action='store_true', default= False, help='visualize graph')
	args = parser.parse_args()

	# Parsear los argumentos
    if not os.path.exists(args.file):
        print(Fore.RED + "Error. The file dont exist." + Style.RESET_ALL)
        sys.exit(1)




if __name__ == "__main__":
    main()