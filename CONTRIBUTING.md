# Contributing to InferTrain

Thank you for your interest in contributing to InferTrain! Whether you are reporting a bug, proposing a feature, or submitting code, this guide will help you get started.

Please read the [Contributor License Agreement](CLA.md) and the [Apache-2.0 License](LICENSE) before contributing.

## Table of Contents

- [Contributor License Agreement (CLA)](#contributor-license-agreement-cla)
- [Development Priorities](#development-priorities)
- [Development Documentation](#development-documentation)
- [Contribution Workflow](#contribution-workflow)
- [Coding Conventions](#coding-conventions)
- [Commit Conventions](#commit-conventions)
- [Pull Request Checklist](#pull-request-checklist)
- [Reporting Issues](#reporting-issues)
- [Contact](#contact)

## Contributor License Agreement (CLA)

All contributors must sign the [Contributor License Agreement](CLA.md) before their contributions can be merged. The first time you open a Pull Request, [cla-assistant](https://cla-assistant.io) will automatically post a comment with a signing link and add a CLA status check — signing only takes a minute.

The CLA clarifies the intellectual property rights granted with your contributions and is fully consistent with the Apache-2.0 license. By signing, you do not give up ownership of your own code; you only grant the project a perpetual, irrevocable license to distribute your contributions under Apache-2.0.


## Contribution Workflow

1. Fork the repository
2. Create a feature branch
3. Implement the feature
4. Test your changes
5. Commit your changes (see [Commit Conventions](#commit-conventions))
6. Submit a Pull Request

## Coding Conventions

- Ensure no Chinese in code and comments
- Ensure code **strictly does not exceed 80 columns**, including comments

## Commit Conventions

- Ensure each commit is a complete, independent, small change
- Ensure commit messages have no Chinese
- Add `-s` flag when committing to generate Sign-off:

```bash
git commit -s -m "your commit message"
```

## Pull Request Checklist

Before submitting your Pull Request, make sure:

- [ ] You have signed the Contributor License Agreement (required for the first PR)
- [ ] No Chinese in code or comments
- [ ] Code strictly does not exceed 80 columns, including comments
- [ ] Each commit is a complete, independent, small change
- [ ] Commit messages contain no Chinese and are signed off with `-s`

## Reporting Issues

- Search existing issues first to avoid duplicates
- Use a clear and descriptive title
- Provide steps to reproduce, expected behavior, and actual behavior
- Include environment details (OS, architecture, Rust version) when relevant

## Contact

- Issues: GitHub Issues
- Discussion: GitHub Discussions
