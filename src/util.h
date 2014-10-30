#include "stdh.h"

/** Fopen with error if file not found*/
FILE* fopen2(const char* fname, char* mode);

/** Low precision timer class the mimics Matlab tic/toc behavior with accuracy 1-2ms. */
class Timer{
private:
	double startTime;
	char str[256];
public:
	Timer();
	Timer(char* str);
	void tic();	/**< Start timer. */
	void tic(char* str); /**< Start timer with label. */
	double toc(int nrtimes = 1); /**< Print time measurement, optionally divide by nrtimes (when calculating average). */
};

#pragma once

class PerformanceTimer
{
private:
	long long startTime, currentTime, frequency;
	//frequency = performance-counter frequency, in counts per second.

public:
	/*
	Sets the initial time of the timer to the current time of the system.
	*/
	void init();

	/*
	Sets the current time of the timer with the current time of the system.
	*/
	void takeTime();

	/*
	Gets the elapsed time in seconds; the precision is in miliseconds.
	Resets the initial time to the current time of the system.
	*/
	float getElapsedTimeAndReInit();

	/*
	Gets the elapsed time in seconds; the precision is in miliseconds.
	*/
	float getElapsedTime();

	PerformanceTimer(void);
	~PerformanceTimer(void);
};
