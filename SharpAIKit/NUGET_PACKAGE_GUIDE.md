# NuGet Package Publishing Guide

## Package Information

- **Package ID**: SharpAIKit
- **Version**: 0.1.0
- **Authors**: Dustin Dong
- **Company**: SharpAIKit
- **License**: MIT
- **Repository**: https://github.com/dxpython/SharpAIKit

## Package Files

After building, you should have:
- `SharpAIKit.0.1.0.nupkg` - Main package
- `SharpAIKit.0.1.0.snupkg` - Symbols package

## Build the Package

```bash
cd src/SharpAIKit
dotnet pack -c Release
```

The packages will be created in:
- `bin/Release/SharpAIKit.0.1.0.nupkg`
- `bin/Release/SharpAIKit.0.1.0.snupkg`

## Verify Package Contents

You can inspect the package using:

```bash
# Extract and view package contents
unzip -l bin/Release/SharpAIKit.0.1.0.nupkg

# Or use NuGet CLI
nuget list SharpAIKit -Source bin/Release
```

## Publish to NuGet.org

### Prerequisites

1. Create a NuGet.org account at https://www.nuget.org
2. Create an API key at https://www.nuget.org/account/apikeys
3. Save your API key securely

### Publish Command

```bash
dotnet nuget push ./bin/Release/SharpAIKit.0.1.0.nupkg -k <YOUR_API_KEY> -s https://api.nuget.org/v3/index.json --skip-duplicate
```

**Important Notes:**
- Replace `<YOUR_API_KEY>` with your actual NuGet API key
- The `--skip-duplicate` flag prevents errors if the version already exists
- The symbols package (`.snupkg`) will be automatically published if found

### Alternative: Publish Both Packages Separately

```bash
# Publish main package
dotnet nuget push ./bin/Release/SharpAIKit.0.1.0.nupkg -k <YOUR_API_KEY> -s https://api.nuget.org/v3/index.json

# Publish symbols package (optional, for debugging)
dotnet nuget push ./bin/Release/SharpAIKit.0.1.0.snupkg -k <YOUR_API_KEY> -s https://api.nuget.org/v3/index.json
```

## Verify Publication

After publishing, verify the package is available:

1. Visit: https://www.nuget.org/packages/SharpAIKit
2. Check that version 0.1.0 is listed
3. Verify the README.md is displayed correctly
4. Test installation: `dotnet add package SharpAIKit --version 0.1.0`

## Package Contents

The package includes:
- ✅ Main library DLL (`SharpAIKit.dll`)
- ✅ XML documentation (`SharpAIKit.xml`)
- ✅ README.md (displayed on NuGet.org)
- ✅ All dependencies properly referenced
- ✅ Symbols package for debugging

## Troubleshooting

### Package Already Exists
If you get an error about the package already existing:
- Use `--skip-duplicate` flag, or
- Increment the version number in `SharpAIKit.csproj`

### API Key Issues
- Ensure your API key has "Push" permissions
- Check that the API key hasn't expired
- Verify you're using the correct key (not the key name)

### README Not Displaying
- Ensure `README.md` is in the project root
- Check that `<PackageReadmeFile>README.md</PackageReadmeFile>` is in the csproj
- Verify the file is included with `<None Include="README.md" Pack="true" PackagePath="\" />`

## Next Steps After Publishing

1. **Update Documentation**: Update README.md with NuGet package link
2. **Create Release**: Create a GitHub release with release notes
3. **Announce**: Share the package on social media/forums
4. **Monitor**: Watch for issues and user feedback

## Version Management

For future releases:
1. Update `<Version>` in `SharpAIKit.csproj`
2. Update `RELEASE_NOTES.md` with new features
3. Rebuild and publish following the same steps

---

**Good luck with your NuGet package publication! 🚀**

