#ifndef TEST_ARMA_ENVIRONMENT_HH
#define TEST_ARMA_ENVIRONMENT_HH

#include <gtest/gtest.h>

class ARMA_environment: public ::testing::Environment {

public:

	void
	SetUp() override;

	void
	TearDown() override;

};


#endif // vim:filetype=cpp
