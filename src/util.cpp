#include "util.h"

FILE* fopen2(const char* fname, char* mode){
	FILE* f = fopen(fname, mode);
	if (!f){
		printf("File not found: %s\n", fname);
		getch();
		exit(-1);
	}
	return f;
}


#define PRINT_TIME 1

Timer::Timer():Timer(""){	
}
Timer::Timer(char* str2){
	strcpy(str, str2);
	tic();
};
void Timer::tic(){
	startTime = clock();
}
void Timer::tic(char* str2){
	strcpy(str, str2);
	startTime = clock();
}

double Timer::toc(int nrtimes){
	double elapsed = (clock() - startTime) / CLOCKS_PER_SEC;
	elapsed /= nrtimes;
	if (elapsed >= 60.0)
	{
		double minutes = floor(elapsed / 60.0);
		elapsed -= minutes*60.0;
#if PRINT_TIME
		printf("%s Time elapsed %d minutes, %f seconds\n", str, (int)minutes, elapsed);
#endif
	}
	else
#if PRINT_TIME
		printf("%s Time elapsed %f seconds\n", str, elapsed);
#endif
	return elapsed;
}

void getTime(char* ttime){
	char days[7][20] = {
		"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"
	};
	time_t rawtime;
	struct tm * timeinfo;
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	sprintf(ttime, "%04d.%02d.%02d %s %02d:%02d:%02d", 1900 + timeinfo->tm_year, timeinfo->tm_mon + 1, timeinfo->tm_mday,
		days[timeinfo->tm_wday], timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);
}

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <assert.h>

PerformanceTimer::PerformanceTimer(void)
{
	assert(sizeof(LARGE_INTEGER) == sizeof(long long));
}

PerformanceTimer::~PerformanceTimer(void)
{
}

void PerformanceTimer::init()
{
	QueryPerformanceCounter((LARGE_INTEGER*)&startTime);
}

void PerformanceTimer::takeTime()
{
	QueryPerformanceCounter((LARGE_INTEGER*)&currentTime);
	QueryPerformanceFrequency((LARGE_INTEGER*)&frequency);
}

float PerformanceTimer::getElapsedTime()
{
	//the time is in milliseconds
	return (float)(1000 * (
		((LARGE_INTEGER*)&currentTime)->QuadPart -
		((LARGE_INTEGER*)&startTime)->QuadPart + 0.0) /
		((LARGE_INTEGER*)&frequency)->QuadPart);
}

float PerformanceTimer::getElapsedTimeAndReInit()
{
	takeTime();
	float elapsedTime = getElapsedTime();
	init();

	return elapsedTime;
}