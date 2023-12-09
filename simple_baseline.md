## Description:

The key idea is that sentences containing important information are likely to be connected to other important sentences. By iteratively computing scores based on these connections, TextRank identifies the most important sentences for summarization. TextRank leverages cosine similarity to establish connections between sentences, and then the PageRank algorithm to rank the importance of these sentences. Finally, we return the k most important sentences as the output summary. 

## How to run the simple baseline:

python3 simple_baseline.py

## Sample Input:

Input Text:

I enjoy vintage books and movies so I enjoyed reading this book.  The plot was unusual.  Don't think killing someone in self-defense but leaving the scene and the body without notifying the police or hitting someone in the jaw to knock them out would wash today.Still it was a good read for me.


Output Summary: 

 I enjoy vintage books and movies so I enjoyed reading this book.  

Gold Standard Summary: 

Nice vintage story

Input Text: 
 
 This book is a reissue of an old one; the author was born in 1910. It's of the era of, say, Nero Wolfe. The introduction was quite interesting, explaining who the author was and why he's been forgotten; I'd never heard of him.The language is a little dated at times, like calling a gun a &#34;heater.&#34;  I also made good use of my Fire's dictionary to look up words like &#34;deshabille&#34; and &#34;Canarsie.&#34; Still, it was well worth a look-see.

Output Summary:

 The language is a little dated at times, like calling a gun a &#34;heater.&#34;  I also made good use of my Fire's dictionary to look up words like &#34;deshabille&#34; and &#34;Canarsie.&#34; Still, it was well worth a look-see.

Gold Standard Summary:

Oldie

Input Text:

A beautiful in-depth character description makes it like a fast pacing movie. It is a pity Mr Merwin did not write 30 instead only 3 of the Amy Brewster mysteries.

Output Summary:

It is a pity Mr Merwin did not write 30 instead only 3 of the Amy Brewster mysteries.

Gold Standard Summary:

Nice old fashioned story

## Sample Output Score:
ROUGE-N SCORE: 0.13333333333333333


