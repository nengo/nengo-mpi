Overview
--------
[ezProgressBar][1] is now 3 C++ progress classes with different output styles (ezProgressBar, ezETAProgressBar and ezRateProgressBar).

They've been compiled and tested with the latest Ubuntu Linux (g++) and Cygwin (g++) in MS Windows 7.

[1]: https://sourceforge.net/projects/ezprogressbar/

<hr>
Features
--------
-   Prints either on one line up to 52 characters (ezProgressBar) or custom width (ezETAProgressBar).
-   Time average performance information (ezRateProgressBar).
-   Only depends on the standard iostream class.
-   Tiny download and no linking due to single header file implementations.
-   MIT license.
-   Minimal learning curve due to examples.
-   Regression tested and memory tested with valgrind.

<hr>
Examples
--------
###ezProgressBar
ezProgressBar is an easy-to-use C++ progress bar class that efficiently prints a single line without carriage returns, so it's ideal for programs that redirect output to a file (otherwise your file would be polluted with numerous lines of progress). It uses a single header file and only depends on the standard iostream. Each dot appears at 2.5 percent intervals, and there are no strings before or after it so you can customize it with labels and messages when it's done. This was inspired by the progress message in the GDAL GIS utilities. It looks like this when complete:

<pre><code>
0...10...20...30...40...50...60...70...80...90...100
</code></pre>

This would create the example:

    #include "ezProgressBar.hpp"
    
    #ifdef WIN32
      #include <windows.h>
    #else
      #include <unistd.h>
    #endif

    int main() {
      int n = 100;
      ez::ezProgressBar p(n);
      p.start();
      
      for(int i=0; i <= n; ++i, ++p) {
        #ifdef WIN32
          Sleep(1000);
        #else
          sleep(1);
        #endif
      }
      
      return 0;
    }

###ezETAProgressBar
ezETAProgressBar is a more traditional growing progress bar that reports a percentage and a time-remaining estimate, from the moment it was invoked. The bar fits on one-line, but uses an overwriting carriage-return to reprint itself. This is how it looks:

<pre><code>
 90% [################################################      ] ETA 1d 3h 46m 34s
</code></pre>

This would create the example:

    #include "ezETAProgressBar.hpp"
    
    #ifdef WIN32
      #include <windows.h>
    #else
      #include <unistd.h>
    #endif

    int main() {
      int n = 10;
      ez::ezETAProgressBar p(n);
      p.start();
      
      for(int i=0; i <= n; ++i, ++p) {
        #ifdef WIN32
          Sleep(1000);
        #else
          sleep(1);
        #endif
      }
      
      return 0;
    }
    
###ezRateProgressBar
The most advanced and informative is ezRateProgressBar. It reports percentage complete, the time that has elapsed and estimated remainder, plus how many tasks have been done and remain. The rate reported is just an average. The units label can be defined in your code. It looks like this:

<pre><code>
Done |  Elapsed | Remaining | Processed | Unprocessed | Rate
 40% | 00:04:00 |  00:06:00 |  4,000 MB |    6,000 MB | 17 MB/s 
</code></pre>

This would create the example:

    #include "ezRateProgressBar.hpp"
    
    #ifdef WIN32
      #include <windows.h>
    #else
      #include <unistd.h>
    #endif

    int main() {
      int n = 1000;
      ez::ezRateProgressBar<int> p(n);
      p.units = "MB";
      p.start();
      
      for(int i=0; i <= n; i += 100) {
        p.update(i);
        #ifdef WIN32
          Sleep(1000);
        #else
          sleep(1);
        #endif
      }
      
      return 0;
    }
    
<hr>
Download
--------
[Source Code, Examples and Tests](http://sourceforge.net/projects/ezprogressbar/files/)

<hr>
Testing
-------
    make

<hr>
Installation
------------
    sudo make install PREFIX=/usr/local
   
<hr>
Distribution
------------
    make html
    make clean
    make dist VER=2.1.0

<hr>
Publishing
----------
    ssh -t rsz,ezprogressbar@shell.sourceforge.net create
    scp html/*.html ez*Bar.hpp rsz,ezprogressbar@shell.sourceforge.net:/home/project-web/ezprogressbar/htdocs

<hr>
Changelog
---------
2.1.1 20120630

-   Added syntax highlighting to markdown output.
-   Fixed licenses in files.

<hr>
License
-------
Copyright (C) 2011,2012 Remik Ziemlinski. See MIT-LICENSE.

<a href="http://sourceforge.net">
<img src="http://sourceforge.net/sflogo.php?group_id=545202&amp;type=2" 
width="125" height="37" border="0" alt="SourceForge.net Logo" /></a>

<link rel="stylesheet" href="http://yandex.st/highlightjs/7.0/styles/default.min.css">
<script src="http://yandex.st/highlightjs/7.0/highlight.min.js"></script>
<script>hljs.initHighlightingOnLoad();</script>