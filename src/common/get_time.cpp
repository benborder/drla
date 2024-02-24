#include "get_time.h"

#include <cassert>
#include <ctime>
#include <fstream>

std::string drla::get_time()
{
	time_t rawtime = 0;
	time(&rawtime);
	struct tm* timeinfo = localtime(&rawtime);

	std::string buffer;
	buffer.resize(16);
	strftime(buffer.data(), buffer.size(), "%Y%m%dT%H%M%S", timeinfo);
	return buffer.c_str();
}
