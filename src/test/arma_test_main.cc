#include "arma_environment.hh"

int
main(int argc, char* argv[]) {
	::testing::InitGoogleTest(&argc, argv);
	::testing::AddGlobalTestEnvironment(new ARMA_environment);
	return RUN_ALL_TESTS();
}

