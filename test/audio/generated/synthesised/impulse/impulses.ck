// this file is used to generate impulses at every second for a duration of 10 seconds
// to run do: chuck -r44100 impulses.ck
Impulse imp => WvOut file => blackhole;
(second/samp) $ int => int sr;
"impulses_1second_" + Std.itoa(sr) + ".wav" => file.wavFilename;
for (int i; i < 10; i++)
{
 // 1::second => now; // skip 1::second from beginning of file
  1=> imp.next;
  1::samp => now;
}
file.closeFile();
