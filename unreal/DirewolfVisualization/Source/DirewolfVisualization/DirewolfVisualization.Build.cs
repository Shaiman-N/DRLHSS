// Copyright Epic Games, Inc. All Rights Reserved.
// DIREWOLF Visualization - Unreal Engine Build Configuration

using UnrealBuildTool;

public class DirewolfVisualization : ModuleRules
{
	public DirewolfVisualization(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
	
		PublicDependencyModuleNames.AddRange(new string[] { 
			"Core", 
			"CoreUObject", 
			"Engine", 
			"InputCore",
			"Niagara",
			"MovieSceneCapture",
			"LevelSequence",
			"MovieScene",
			"CinematicCamera",
			"Json",
			"JsonUtilities",
			"HTTP",
			"Sockets",
			"Networking"
		});

		PrivateDependencyModuleNames.AddRange(new string[] { 
			"Slate",
			"SlateCore"
		});

		// VR Support (Optional)
		if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			PrivateDependencyModuleNames.AddRange(new string[] {
				"HeadMountedDisplay",
				"SteamVR"
			});
		}
	}
}
