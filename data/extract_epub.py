"""Extract list of epubs to a text file"""

from __future__ import print_function
from codecs import open
import subprocess
import os
from pipes import quote
from natsort import natsorted, as_utf8
import re

def extract_epubs(epub_list):
	outfile = "./data.txt"
	open(outfile, "w").close()
	outfile_f = open(outfile, "w", "utf-8")

	temp_file = "./temp.txt"

	all_data = []
	for epub in epub_list:
		open(temp_file, "w").close()
		cmd = "ebook-convert %s %s" % (quote(epub), quote(temp_file))
		p = subprocess.Popen(cmd, shell=True)
		p.communicate()

		with open(temp_file, "r", "utf-8") as f:
			data = f.read()
			data = re.sub(r"\n{3,}", "\n\n", data)
			outfile_f.write(data)

		outfile_f.write("\n")

	os.remove(temp_file)
	outfile_f.close()


if __name__ == "__main__":
	epub_list = [os.path.join(".", file) for file in os.listdir(".") if file.endswith(".epub")]
	epub_list = natsorted(epub_list, key=as_utf8)
	extract_epubs(epub_list)
