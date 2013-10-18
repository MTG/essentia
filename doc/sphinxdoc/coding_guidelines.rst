.. highlight:: cpp

General development guidelines
==============================

Fixing bugs
-----------

**IMPORTANT:** when fixing bugs, try to always fix the real cause of the problem, and do
not be satisfied with a workaround as it will backfire some time later when you expect it the least.
Even if really fixing a bug needs 3x more time than a simple workaround, DO IT! If you don't
know how to fix the cause of a bug instead of just hiding it, seek help!


Doing commits in Github
-----------------------

* as much as possible, do commits that are one semantic action: e.g., if you add a feature
  that changes 5 files, commit the 5 files together, if you change 2 lines inside the same
  file that do 2 different things, then commit it in 2 parts
* use explicit and useful and descriptive messages. That doesn't prevent you from being
  humorous in them, but messages such as "bsdjfh" are *not* allowed.


Coding Guidelines
=================

Coding style guidelines
-----------------------

Coding style guidelines are here to help readability. Everybody is thus **required** to
follow them. The recommended style is a mix between `boost`_, `Qt`_, and the `STL`_
(probably the 3 most well-designed C++ libraries), but there is no official written
guidelines for it now. Have a look at the examples to get the feeling of it.

Specific points:

* tabs are forbidden, use spaces. Used tabs sizes are 2 spaces in C++, 4 spaces in python.
* a comma (',') is **always** followed by a space (' ')
* curly brackets are NOT on a new line, except (maybe) for function definitions.
* in C++ class declarations, put only one space before the access type (public, protected, private)
* the else statement should go on an extra line
* spacing for binary operators (+, -, /, \*, etc) should make the code as readable as possible

::

  class Test {
   public:
    void foo(int param1, bool param2);

   private:
    void bar() {
      if (true) {
        float f = 42 + 1./16;
      }
      else {
        int n = 521 + 816;
      }
    }
  };


As a rule of thumb: if you can write the same code and use fewer characters while keeping
the code clean, the shorter version is always the better one.
Also think about your fellow programmers who don't have an IDE with autocompletion, nor
a widescreen that spans more than 300 columns, and try to keep your code as short, clean
and concise as possible.

The key word to remember is **CONSISTENCY**. Whenever in doubt, look at some other file 
and follow the rules/styles applied in that file. For instance, ``#ifdef __MYFILE_H__`` 
is *not* correct. If you look at other files, they are all written like this: 
``#ifdef ESSENTIA_MYFILE_H``, so follow that rule.

For Windows users: make sure that you do not use Windows end-of-lines, but Unix ones.
Basically, any editor that is not Notepad should be able to deal with that.


General C++ pitfalls
--------------------

passing arguments as const-refs:

  if you want to pass arguments to a function that are not going to be changed inside that function, use const references. ::

    void f(const string& str); // good

    void f(const string str);  // wrong: makes a copy
    void f(string str);        // wrong: makes a copy
    void f(string& str);       // maybe wrong: if you don't modify it, add the const qualifier


using namespace xxx:
  in headers, it is **strictly** forbidden to have ``using namespace`` directives, however
  you can have them in the .cpp files (it is even recommended).




The ``Real`` data type
----------------------

Use the ``Real`` type when working with real numbers. By default this type is typedef'd to
``float``, but we could just as easily change it to ``double`` for doing precision tests, etc...
Also, when declaring numeric constants, if you get warnings about float/double issues, do
not specify 'f' at the end of the variable, but cast it to Real. ::

  Real pi = 3.14f; // wrong
  Real pi = (Real)3.14;


The ``bool`` data type
----------------------

Use the ``bool`` type when working with booleans. The use of ``int`` is strictly prohibited. ::

  while (1) { do_sth(); } // wrong
  while (true) { do_sth(); } // good


Error handling
--------------

No single function should return error codes. We're programming in C++, the standard way of
signalling an error is to throw an exception. Also, when checking for errors, incorrect
inputs, etc... do it as soon as possible (and not when you need it) and throw an exception.
That means that if we get past this point of execution in the code, all inputs and conditions
are valid.

The const keyword
-----------------

Use ``const`` whenever possible. Do not remove a const qualifier at some place because it
is "easier" to do something, but rather look for which function does not accept a ``const`` and
modify this one (ie: no workaround, solve the real problem!)

Naming conventions
------------------

* Names should not be abbreviated: NoiseGen should be NoiseGenerator, FreqBands should be
  FrequencyBands, etc...
* Anything that has a size should be called xxxxSize. I.e. windowSize, bufferSize, ...
* As your parameters need to be used in Python, there can be no spaces in the names. Also,
  use camelCase with small caps for first character (cutoffFrequency, ...)

Case-sensitivity of the identifiers
-----------------------------------

All identifiers (names of the algorithms in the factory, names of the parameters, etc...)
are case-sensitive. That means that 'Mfcc' != 'mfcc' != 'MFCC' so please make sure you
spell things correctly. To help in this task, there are naming guidelines: classes should
have CamelCaseNames and parameters should have camelCaseNamesWithSmallFirstLetter.


Parsing parameters
------------------

Parameters should never be parsed in the ``compute()`` method, but rather in the
``configure()`` method. If needed, create a special member variable (protected) that you
will need to store the result of parsing your parameter.

The reason behind this is that parameters can only be changed in the ``configure()`` method,
but then could be parsed a lot of (unnecessary) times in the ``compute()`` method. Thus it is
much more efficient to parse them once and for all in the ``configure()`` method.

On the use of generic/specific types
------------------------------------

(``Essentia::Spectrum`` vs. ``std::vector<float>``)

Again, this is not a rule, but a guideline, however it would be really nice (and also useful)
if everyone were to follow it. The idea is to use the most generic types whenever we can,
instead of specialized types that may reveal to be too specialized afterwards.

That's more or less the frame of mind when you're working with Matlab for instance, where you
only work with arrays and matrices, and not with Spectrums, LPCs, IIR filter coeffs, etc...

We feel it is up to the person doing the computations to make sure they're not feeding stupid
data to the algorithms, but it also allows them to do experiments very quickly (not having to
have wrappers for each and every single type) and have more generic algorithms that can be
applied to a broader range of problems.

Parameters versus inputs
------------------------

* Algorithms can have input/output-sizes as parameters, but if an input is given with a
  different size, it should not complain and re-initialize itself.
* If algorithms are 'generators', 'outputSizes' (etc) should be parameters.
* Inputs should be called either "array" (generic type), "signal" (audio/envelope/... signal)
  or "spectrum" (...) unless there is a good reason not to use these names.
* Outputs should either use the name of the algorithm, or if needed, something more meaningful.

Things to watch out for
-----------------------

* Make sure your algorithm doesn't generate NaN's nor INF's.
* Make sure your algorithm returns results which are meaningful. Make sure results are as
  little as possible dependent on the blocksize. For example, spectral centroid doesn't
  return a bin number, it returns a frequency!

Error checking
--------------

* Both ``configure()`` and ``compute()`` should use ``EssentiaException`` wherever possible
* Unit tests should be written for each algorithm.
* You should write at least one function which takes a filename as input (wav) and
  generates output as a unit-test.
* All algorithms need to be peer-reviewed.


.. _boost: http://www.boost.org/
.. _Qt: http://qt.digia.com/
.. _STL: http://www.sgi.com/tech/stl/
