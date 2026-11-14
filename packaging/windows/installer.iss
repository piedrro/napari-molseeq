[Setup]
AppId={{6F842D6D-E1AD-4F2B-9B1D-3CFCA94AFDB4}
AppName=MolSeeq
AppVersion={#AppVersion}
AppPublisher=MolSeeq
AppPublisherURL=https://github.com/piedrro/napari-molseeq
DefaultDirName={localappdata}\Programs\napari-molseeq
DefaultGroupName=MolSeeq
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir={#OutputDir}
OutputBaseFilename=napari-molseeq-Setup-{#AppVersion}-x64
SetupIconFile={#BundleDir}\_internal\napari\resources\icon.ico
UninstallDisplayIcon={app}\napari-molseeq.exe
LicenseFile=..\..\LICENSE
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; Replace the bundled runtime on upgrade. Logs and caches are stored outside it.
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{group}\MolSeeq"; Filename: "{app}\napari-molseeq.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\MolSeeq"; Filename: "{app}\napari-molseeq.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\napari-molseeq.exe"; Description: "Launch MolSeeq"; Flags: nowait postinstall skipifsilent
