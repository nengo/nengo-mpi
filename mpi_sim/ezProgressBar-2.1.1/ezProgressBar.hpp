/*
Copyright (C) 2011,2012 Remik Ziemlinski. See MIT-LICENSE.

CHANGELOG

v0.0.0 20110502 rsz Created.
v2.0.1 20111006 rsz Added default constructor value.
*/

#ifndef EZ_PROGRESSBAR_H
#define EZ_PROGRESSBAR_H

#include <iostream>

namespace ez {
// One-line minimally printing progress bar inspired by GDAL.
// NOTE: Does not print new-line after 100.
class ezProgressBar {
public:
	ezProgressBar(unsigned int _n=0) : n(_n), pct(0), cur(0) {}
	void reset() { pct = 0; cur = 0; }
	void start() { std::cout << '0'; std::cout.flush(); }
	void operator++() {
		if (cur >= n) return;
		++cur;
		
		setPct( (float)cur/n );
	};
	
	// Set 0.0-1.0, where 1.0 equals 100%.
	void setPct(float Pct) {
		short delta = (short)(Pct*1000 - pct);
		if (delta < 25) return;
		
		do {
			pct += 25;
			if ( (pct % 100) == 0 ) 
				std::cout << pct/10;
			else
				std::cout << '.';
		} while((delta -= 25) >= 25);
		std::cout.flush();
	};

	unsigned int n;
	unsigned int cur;
	unsigned short pct; // Stored as 0-1000, so 2.5% is encoded as 25.
};
}
#endif // EZ_PROGRESSBAR_H
